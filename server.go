package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"granger-test/internal/granger"
)

func runServe(args []string) {
	addr := ":8080"
	for i := 0; i < len(args); i++ {
		if (args[i] == "--addr" || args[i] == "-addr") && i+1 < len(args) {
			addr = args[i+1]
			i++
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/api/test", handleGrangerTest)

	fmt.Printf("granger-test serving on %s\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintf(os.Stderr, "serve: %v\n", err)
		os.Exit(1)
	}
}

func handleGrangerTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		X   []float64 `json:"x"`
		Y   []float64 `json:"y"`
		Lag int       `json:"lag"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(req.X) == 0 || len(req.Y) == 0 {
		http.Error(w, "x and y arrays required", http.StatusBadRequest)
		return
	}
	if req.Lag <= 0 {
		http.Error(w, "lag must be positive", http.StatusBadRequest)
		return
	}

	res, err := granger.Test(req.X, req.Y, req.Lag)
	if err != nil {
		http.Error(w, "test: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"f_x_to_y":  res.FX,
		"f_y_to_x":  res.FY,
		"p_x_to_y":  res.PX,
		"p_y_to_x":  res.PY,
		"direction":  res.Direction,
	})
}
