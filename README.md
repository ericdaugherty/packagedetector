# Package Detector
[![Go Report Card](https://goreportcard.com/badge/github.com/ericdaugherty/packagedetector)](https://goreportcard.com/report/github.com/ericdaugherty/packagedetector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ericdaugherty/packagedetector/blob/master/LICENSE)

This app takes snapshots off a camera (assuming it support HTTP GET snapshots) and uploads
them to Google Vision to determine if there is a package present. If so, an email or Firebase Cloud Message Notification is sent.
