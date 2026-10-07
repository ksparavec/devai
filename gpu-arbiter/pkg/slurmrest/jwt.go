// Package slurmrest is devai's one client for slurmrestd (docs/slurm.md
// Sec. 7 and 9; platform-rebuild C3, C7): an HS256 JWT signer and the
// v0.0.42 calls the router, devai-jobs and devai-control make. Field and
// path names follow the spec this slurmrestd (Debian's 24.11.5) prints,
// stored as deploy/slurm/openapi-v0.0.42.json and checked by the tests.
package slurmrest

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// TokenTTL is the lifetime of a signed token (C3: exp = iat + 300).
const TokenTTL = 300 * time.Second

// Claims are the token's payload: Slurm's auth/jwt reads `sun` (the Slurm
// user name) and `exp`.
type Claims struct {
	Sun string `json:"sun"`
	Iat int64  `json:"iat"`
	Exp int64  `json:"exp"`
}

var b64 = base64.RawURLEncoding

// SignJWT returns an HS256 token {"sun":user, "iat":now, "exp":now+ttl}
// signed with key (the shared jwt_hs256.key, C7).
func SignJWT(key []byte, user string, now time.Time, ttl time.Duration) (string, error) {
	if len(key) == 0 {
		return "", errors.New("jwt: empty key")
	}
	if user == "" {
		return "", errors.New("jwt: empty user")
	}
	header, _ := json.Marshal(map[string]string{"alg": "HS256", "typ": "JWT"})
	claims, err := json.Marshal(Claims{Sun: user, Iat: now.Unix(), Exp: now.Add(ttl).Unix()})
	if err != nil {
		return "", err
	}
	signing := b64.EncodeToString(header) + "." + b64.EncodeToString(claims)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(signing))
	return signing + "." + b64.EncodeToString(mac.Sum(nil)), nil
}

// VerifyJWT checks an HS256 token's signature and expiry and returns its
// claims. The client never needs it; the tests and the fake slurmrestd do.
func VerifyJWT(key []byte, token string, now time.Time) (Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return Claims{}, errors.New("jwt: not three parts")
	}
	var header map[string]string
	hb, err := b64.DecodeString(parts[0])
	if err != nil || json.Unmarshal(hb, &header) != nil || header["alg"] != "HS256" {
		return Claims{}, errors.New("jwt: bad header")
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(parts[0] + "." + parts[1]))
	sig, err := b64.DecodeString(parts[2])
	if err != nil || !hmac.Equal(sig, mac.Sum(nil)) {
		return Claims{}, errors.New("jwt: bad signature")
	}
	var c Claims
	cb, err := b64.DecodeString(parts[1])
	if err != nil || json.Unmarshal(cb, &c) != nil {
		return Claims{}, errors.New("jwt: bad claims")
	}
	if now.Unix() >= c.Exp {
		return Claims{}, fmt.Errorf("jwt: expired at %d", c.Exp)
	}
	return c, nil
}
