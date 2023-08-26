import { request } from './request';

export function get() {
  return request("get", `/`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function options() {
  return request("options", `/`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function routesGet() {
  return request("get", `/routes`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function routesOptions() {
  return request("options", `/routes`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function logGet() {
  return request("get", `/log`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function logPost() {
  return request("post", `/log`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function logOptions() {
  return request("options", `/log`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function runGet() {
  return request("get", `/run`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function runPost() {
  return request("post", `/run`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function runOptions() {
  return request("options", `/run`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function fitPost() {
  return request("post", `/fit`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function fitOptions() {
  return request("options", `/fit`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function loginSuccessGet() {
  return request("get", `/login-success`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function loginSuccessOptions() {
  return request("options", `/login-success`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function loginDexcomGet() {
  return request("get", `/login-dexcom`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

export function loginDexcomOptions() {
  return request("options", `/login-dexcom`, { "header": { "accept": "application/json", "Content-Type": "application/json", }, })();
}

