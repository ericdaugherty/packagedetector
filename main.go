package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"time"

	motion "github.com/ericdaugherty/unifi-nvr-motiondetection"
	gomail "gopkg.in/gomail.v2"
)

var imageURL string
var rect string
var interval int
var motionLogPath string
var cameraID string
var gAuthJSONPath string
var gVisionURL string
var cacheImage string
var notifyClass string
var sleepHour int
var wakeHour int

var lastEmailSent time.Time
var emailFrom string
var emailTo string
var emailServer string
var emailPort int
var emailUser string
var emailPass string
var emailMuteDuration int
var emailOnStartup bool

var r image.Rectangle
var buf bytes.Buffer

type visionRequest struct {
	Payload struct {
		Image struct {
			ImageBytes string `json:"imageBytes"`
		} `json:"image"`
	} `json:"payload"`
}

type visionResponse struct {
	Payload []struct {
		Classification struct {
			Score float64 `json:"score"`
		} `json:"classification"`
		DisplayName string `json:"displayName"`
	} `json:"payload"`
}

func init() {
	flag.StringVar(&imageURL, "imageURL", "", "The URL of the image to fetch.")
	flag.StringVar(&rect, "rect", "", "The x/y coordinates that define the rectangle to use to crop in the form of x,y,x,y")
	flag.IntVar(&interval, "interval", 0, "The interval, in minutes, between executions.")
	flag.StringVar(&motionLogPath, "motionLog", "", "The path to the NVR motion.log file.")
	flag.StringVar(&cameraID, "cameraID", "", "The camera ID of the camera to monitor for motion.")
	flag.StringVar(&gAuthJSONPath, "gAuthJSON", "", "Path to the Google Cloud JSON Auth file.")
	flag.StringVar(&gVisionURL, "gVisionURL", "", "Google Service URL to query for image evaluation.")
	flag.StringVar(&cacheImage, "cacheImage", "img.jpg", "The current image will be written to disk as this filename.")
	flag.StringVar(&notifyClass, "notifyClass", "package", "If the image label matches this value, a notification will be sent.")
	flag.IntVar(&sleepHour, "sleepHour", 0, "The hour to pause image capture (0-23)")
	flag.IntVar(&wakeHour, "wakeHour", 0, "The hour to resume image capture (0-23)")
	flag.StringVar(&emailFrom, "emailFrom", "", "The email address to use for the FROM setting.")
	flag.StringVar(&emailTo, "emailTo", "", "The email address to use for the TO setting.")
	flag.StringVar(&emailServer, "emailServer", "", "The SMTP Server to use to send the email.")
	flag.IntVar(&emailPort, "emailServerPort", 587, "The port to use to connect to the SMTP Server")
	flag.StringVar(&emailUser, "emailUser", "", "The SMTP Username to use, if needed.")
	flag.StringVar(&emailPass, "emailPass", "", "The SMTP Password to use, if needed.")
	flag.IntVar(&emailMuteDuration, "emailMuteMinutes", 60, "The amount of time to wait between sending emails.")
	flag.BoolVar(&emailOnStartup, "emailOnStart", false, "Send an email on startup when this flag is present.")
}

func main() {

	flag.Parse()

	if imageURL == "" || gAuthJSONPath == "" || gVisionURL == "" {
		flag.Usage()
		os.Exit(1)
	}

	if rect != "" {
		rpoints := strings.Split(rect, ",")
		if len(rpoints) != 4 {
			log.Fatal("rect flag not in the form of x,y,x,y")
		}
		var ripoints [4]int
		for i, num := range rpoints {
			v, err := strconv.Atoi(num)
			if err != nil {
				log.Fatal("Error converting number " + num + " to int.")
			}
			ripoints[i] = v
		}
		r = image.Rect(ripoints[0], ripoints[1], ripoints[2], ripoints[3])
	}

	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer func() {
		signal.Stop(c)
		cancel()
	}()

	go func() {
		select {
		case <-c:
			cancel()
		}
	}()

	// Check the image before starting our monitoring/loop and force email if specified.
	processImage(emailOnStartup)

	// If the motionLogPath is set,
	if motionLogPath != "" {
		if cameraID == "" {
			log.Fatal("The cameraID parameter must be set if the motionLogPath is present.")
		}

		md, err := motion.DetectMotion(motionLogPath)
		if err != nil {
			log.Fatal("Unable to open the motion.log file: "+motionLogPath, err.Error())
		}
		md.AddStopMotionCallback(cameraID, func(string, string) {
			if isAwake(time.Now(), sleepHour, wakeHour) {
				processImage(false)
			}
		})
	}

	log.Println("Running...")

	if interval > 0 {
		ticker := time.NewTicker(time.Duration(interval) * time.Minute)
		for {
			select {
			case <-ticker.C:
				if isAwake(time.Now(), sleepHour, wakeHour) {
					processImage(false)
				}
			case <-ctx.Done():
				return
			}
		}
	} else {
		select {
		case <-ctx.Done():
			return
		}
	}
}

func processImage(forceEmail bool) {
	token, err := getGoogleToken()
	if err != nil {
		log.Println("Unable to get Google Cloud Token. Error:", err.Error(), "StdOut:", token)
		return
	}

	err = fetchImage()
	if err != nil {
		log.Println("Error fetching image.", err.Error())
		return
	}

	if rect != "" {
		err = cropImage(r)
	}
	if err != nil {
		log.Println("Crop failed.", err.Error())
		return
	}

	resp, err := evaluateImageJSON(token)
	if err != nil {
		log.Println("Error evaluating image.", err.Error())
		return
	}

	for _, p := range resp.Payload {
		log.Printf("Result: %v, Confidence: %f\n", p.DisplayName, p.Classification.Score)
		if p.DisplayName == notifyClass || forceEmail {
			if time.Now().After(lastEmailSent.Add(time.Duration(emailMuteDuration) * time.Minute)) {
				emailResult("Package Received!", fmt.Sprintf("A package has been identified at the door with %f certainty", p.Classification.Score))
			}
		}
	}
}

func fetchImage() error {
	url := imageURL
	buf.Reset()

	response, err := http.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	_, err = io.Copy(&buf, response.Body)
	if err != nil {
		return err
	}

	return nil
}

func cropImage(r image.Rectangle) error {

	srcimg, _, err := image.Decode(&buf)
	if err != nil {
		return err
	}

	memimg := image.NewRGBA(srcimg.Bounds())

	draw.Draw(memimg, memimg.Bounds(), srcimg, image.Point{0, 0}, draw.Src)
	newimg := memimg.SubImage(r)

	buf.Reset()
	err = jpeg.Encode(&buf, newimg, nil)

	return err
}

func evaluateImageJSON(token string) (visionResponse, error) {

	b := buf.Bytes()

	request := &visionRequest{}
	request.Payload.Image.ImageBytes = base64.StdEncoding.EncodeToString(b)
	reqBytes, err := json.Marshal(request)
	if err != nil {
		return visionResponse{}, err
	}

	req, err := http.NewRequest("POST", gVisionURL, bytes.NewBuffer(reqBytes))
	if err != nil {
		return visionResponse{}, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return visionResponse{}, err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	var response visionResponse
	err = json.Unmarshal(body, &response)
	if err != nil {
		return response, err
	}

	if cacheImage != "" {
		ioutil.WriteFile(cacheImage, b, 0644)
	}

	return response, nil
}

func getGoogleToken() (string, error) {
	// Get Bearer Token
	cmd := exec.Command("gcloud", "auth", "application-default", "print-access-token")
	cmd.Env = append(os.Environ(),
		"GOOGLE_APPLICATION_CREDENTIALS="+gAuthJSONPath,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}

	return strings.TrimSpace(string(out)), nil
}

func emailResult(subject string, body string) {
	m := gomail.NewMessage()
	m.SetHeader("From", emailFrom)
	m.SetHeader("To", emailTo)
	m.SetHeader("Subject", subject)
	m.SetBody("text/html", body)
	m.Embed(cacheImage)

	d := gomail.NewDialer(emailServer, emailPort, emailUser, emailPass)

	if err := d.DialAndSend(m); err != nil {
		log.Println(err)
		return
	}
	lastEmailSent = time.Now()
}

func isAwake(now time.Time, sleepHour int, wakeHour int) bool {
	hour := now.Hour()
	if (sleepHour > wakeHour) && (hour < sleepHour && hour >= wakeHour) {
		return true
	} else if (sleepHour < wakeHour) && (hour < sleepHour || hour >= wakeHour) { // Ex sleep 1, wake 7
		return true
	} else if sleepHour == wakeHour { // Both the same, so don't sleep.
		return true
	}
	return false
}
