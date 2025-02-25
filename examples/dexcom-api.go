package main

import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "net/url"
    "strings"
)

const (
    clientID     = "your_client_id"
    clientSecret = "your_client_secret"
    redirectURI  = "http://localhost:8080/callback"
)

func main() {
    http.HandleFunc("/", home)
    http.HandleFunc("/callback", callback)
    http.ListenAndServe(":8080", nil)
}

func home(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, `<a href="http://localhost:8080/login" target="_blank">Login with Dexcom</a>`)
}

func callback(w http.ResponseWriter, r *http.Request) {
    authCode := r.URL.Query().Get("code")
    tokenURL := "https://api.dexcom.com/v2/oauth2/token"
    data := url.Values{
        "client_id":     {clientID},
        "client_secret": {clientSecret},
        "code":          {authCode},
        "grant_type":    {"authorization_code"},
        "redirect_uri":  {redirectURI},
    }
    response, err := http.Post(tokenURL, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer response.Body.Close()

    body, err := ioutil.ReadAll(response.Body)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    var result map[string]interface{}
    json.Unmarshal(body, &result)
    access_token := result["access_token"].(string)
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"access_token": access_token})
}
