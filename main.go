package main

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	_ "github.com/go-sql-driver/mysql"
)

var db *sql.DB

func main() {
	var err error
	db, err = sql.Open("mysql", "user:password@tcp(127.0.0.1:3306)/sketch_db")
	if err != nil {
		panic(err)
	}

	http.HandleFunc("/upload-sketch", handleSketchUpload)
	http.ListenAndServe(":8080", nil)
}

type SketchRequest struct {
	Sketch string `json:"sketch"`
}

type SketchResponse struct {
	GeneratedImage string `json:"generated_image"`
}

func handleSketchUpload(w http.ResponseWriter, r *http.Request) {
	var req SketchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid input", http.StatusBadRequest)
		return
	}

	// Decode base64 drawing
	sketchData := req.Sketch[strings.IndexByte(req.Sketch, ',')+1:]
	sketchBytes, _ := base64.StdEncoding.DecodeString(sketchData)

	// Post drawing
	resp, err := http.Post("http://model-server-url/generate", "application/octet-stream", strings.NewReader(string(sketchBytes)))
	if err != nil {
		http.Error(w, "Model server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Get generated
	genImgBytes := make([]byte, resp.ContentLength)
	resp.Body.Read(genImgBytes)
	encodedGenImage := base64.StdEncoding.EncodeToString(genImgBytes)

	// Storing
	_, err = db.Exec("INSERT INTO images (sketch, generated) VALUES (?, ?)", sketchData, encodedGenImage)
	if err != nil {
		http.Error(w, "Database error", http.StatusInternalServerError)
		return
	}

	// Send the generated image back to the frontend
	res := SketchResponse{GeneratedImage: encodedGenImage}
	json.NewEncoder(w).Encode(res)
}
