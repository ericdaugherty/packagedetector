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
	"path"
	"strings"
	"time"

	motion "github.com/ericdaugherty/unifi-nvr-motiondetection"
	gomail "gopkg.in/gomail.v2"
	yaml "gopkg.in/yaml.v2"
)

var configPath string

var config *configuration
var r image.Rectangle
var buf bytes.Buffer
var lastNotificationSent time.Time

type configuration struct {
	Run     runCfg
	Motion  motionCfg
	Image   imageCfg
	Vision  visionCfg
	Email   emailCfg
	WebHook webhookCfg
}

type runCfg struct {
	Interval          int
	SleepHour         int
	WakeHour          int
	NotifyOnStart     bool
	NotifyMuteMinutes int
}

type imageCfg struct {
	URL         string
	CacheFile   string
	ArchivePath string
	Rect        rectCfg
}

type rectCfg struct {
	X1 int
	Y1 int
	X2 int
	Y2 int
}

type motionCfg struct {
	LogPath  string
	CameraID string
}

type visionCfg struct {
	AuthFile     string
	URL          string
	PackageLabel string
	Threshold    int
}

type emailCfg struct {
	From   string
	To     []string
	Server string
	Port   int
	User   string
	Pass   string
}

type webhookCfg struct {
	URL string
}

func (r runCfg) run(ctx context.Context) {
	if r.Interval > 0 {
		ticker := time.NewTicker(time.Duration(r.Interval) * time.Minute)
		for {
			select {
			case <-ticker.C:
				if r.isAwake() {
					processImage(ctx, false)
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

func (r runCfg) isAwake() bool {
	now := time.Now()
	hour := now.Hour()

	if (r.SleepHour > r.WakeHour) && (hour < r.SleepHour && hour >= r.WakeHour) {
		return true
	} else if (r.SleepHour < r.WakeHour) && (hour < r.SleepHour || hour >= r.WakeHour) { // Ex sleep 1, wake 7
		return true
	} else if r.SleepHour == r.WakeHour { // Both the same, so don't sleep.
		return true
	}
	return false
}

func (i imageCfg) isCropSpecified() bool {
	return i.Rect.X1 > 0 ||
		i.Rect.Y1 > 0 ||
		i.Rect.X2 > 0 ||
		i.Rect.Y2 > 0
}

func (i imageCfg) initialize() {
	if i.CacheFile == "" {
		i.CacheFile = "./img.jpg"
	}

	if i.ArchivePath != "" {
		if _, err := os.Stat(i.ArchivePath); os.IsNotExist(err) {
			log.Fatalln("The specified ArchivePath does not exist.", i.ArchivePath)
		}
	}

	if i.URL == "" {
		log.Fatalln("Please specify a valid imageURL in the configuration file.")
	}

	// Check if Rect is specified. If so, setup image.Rectangle.
	if i.isCropSpecified() {
		r = image.Rect(i.Rect.X1, i.Rect.Y1, i.Rect.X2, i.Rect.Y2)
	}
}

func (m motionCfg) initialize(ctx context.Context) {
	if m.LogPath != "" {
		if m.CameraID == "" {
			log.Fatal("The cameraID parameter must be set if the motionLogPath is present.")
		}

		md, err := motion.DetectMotion(m.LogPath)
		if err != nil {
			log.Fatal("Unable to open the motion.log file: "+m.LogPath, err.Error())
		}
		md.AddStopMotionCallback(m.CameraID, func(string, string) {
			if config.Run.isAwake() {
				processImage(ctx, false)
			}
		})
	}
}

func (v visionCfg) initialize() {
	if config.Vision.AuthFile == "" || config.Vision.URL == "" {
		log.Fatalln("Configuration must contain values for vision: authfile and vision: url")
	}
}

func (e emailCfg) initialize() {
	if e.Port == 0 {
		e.Port = 587
	}
}

func (w webhookCfg) initialize() {}

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

type firestorePackage struct {
	Title         string
	Body          string
	Score         float64
	Label         string
	ImageBucket   string
	ImageFilename string
	DateTime      time.Time
}

type webHookBody struct {
	Score float64 `json:"score"`
	Label string  `json:"label"`
	Image string  `json:"image"`
}

func init() {
	flag.StringVar(&configPath, "c", "./pd.yaml", "The path to the config file.")
}

func main() {

	flag.Parse()

	// Load the Configuration
	config = &configuration{}
	b, err := ioutil.ReadFile(configPath)
	if err != nil {
		log.Fatalln("Unable to load configuration file:", configPath, err)
	}
	err = yaml.UnmarshalStrict(b, config)
	if err != nil {
		log.Fatalln("Unable to parse configuration file.", err)
	}

	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)

	config.Image.initialize()

	config.Vision.initialize()

	config.Email.initialize()

	config.WebHook.initialize()

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
	processImage(ctx, config.Run.NotifyOnStart)

	config.Motion.initialize(ctx)

	log.Println("Running...")

	config.Run.run(ctx)
}

func processImage(ctx context.Context, forceNotify bool) {
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

	if config.Image.isCropSpecified() {
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
		log.Printf("Result: %v, Confidence: %f Threshold: .%d\n", p.DisplayName, p.Classification.Score, config.Vision.Threshold)
		if p.DisplayName == config.Vision.PackageLabel && ((p.Classification.Score * 100) > float64(config.Vision.Threshold)) {
			if time.Now().After(lastNotificationSent.Add(time.Duration(config.Run.NotifyMuteMinutes) * time.Minute)) {
				lastNotificationSent = time.Now()
				if config.Email.Server != "" {
					emailResult("Package Received!", fmt.Sprintf("A new package delivery was detected."))
				}
				if config.WebHook.URL != "" {
					sendWebHook(p.Classification.Score, p.DisplayName, buf.Bytes())
				}
			}
		} else if forceNotify {
			if config.Email.Server != "" {
				emailResult("Package Monitor Restarted.", "The Package Monitor server has restarted successfully.")
			}
			if config.WebHook.URL != "" {
				sendWebHook(p.Classification.Score, p.DisplayName, buf.Bytes())
			}
		}
	}
}

func fetchImage() error {
	url := config.Image.URL
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

	req, err := http.NewRequest("POST", config.Vision.URL, bytes.NewBuffer(reqBytes))
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

	ioutil.WriteFile(config.Image.CacheFile, b, 0644)
	if config.Image.ArchivePath != "" {
		ioutil.WriteFile(path.Join(config.Image.ArchivePath, time.Now().Format(time.RFC3339)+".jpeg"), b, 0644)
	}

	return response, nil
}

func getGoogleToken() (string, error) {
	// Get Bearer Token
	cmd := exec.Command("gcloud", "auth", "application-default", "print-access-token")
	cmd.Env = append(os.Environ(),
		"GOOGLE_APPLICATION_CREDENTIALS="+config.Vision.AuthFile,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}

	return strings.TrimSpace(string(out)), nil
}

func emailResult(subject string, body string) {
	e := config.Email
	m := gomail.NewMessage()
	m.SetHeader("From", e.From)
	for _, to := range e.To {
		m.SetHeader("To", to)
	}
	m.SetHeader("Subject", subject)
	m.SetBody("text/html", body)
	m.Embed(config.Image.CacheFile)

	d := gomail.NewDialer(e.Server, e.Port, e.User, e.Pass)

	if err := d.DialAndSend(m); err != nil {
		log.Println(err)
		return
	}
}

func sendWebHook(score float64, label string, image []byte) {

	bodyStruct := &webHookBody{
		Label: label,
		Score: score,
		Image: base64.StdEncoding.EncodeToString(image),
	}

	postBody, err := json.Marshal(bodyStruct)
	if err != nil {
		log.Println("Error marshaling struct into json for POST.")
		return
	}

	req, err := http.NewRequest("POST", config.WebHook.URL, bytes.NewBuffer(postBody))
	if err != nil {
		log.Println("Error creating HTTP POST Request for WebHook.", err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	log.Println("POSTed WebHook, Status: ", resp.Status)
}
