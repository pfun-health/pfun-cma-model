type RequestResult<Data> = Promise<{ response: Response; data: Data; }>;

type GetResult0 = RequestResult<object>;
export function get(): GetResult0;

type OptionsResult0 = RequestResult<object>;
export function options(): OptionsResult0;

type RoutesGetResult0 = RequestResult<object>;
export function routesGet(): RoutesGetResult0;

type RoutesOptionsResult0 = RequestResult<object>;
export function routesOptions(): RoutesOptionsResult0;

type LogGetResult0 = RequestResult<object>;
export function logGet(): LogGetResult0;

type LogPostResult0 = RequestResult<object>;
export function logPost(): LogPostResult0;

type LogOptionsResult0 = RequestResult<object>;
export function logOptions(): LogOptionsResult0;

type RunGetResult0 = RequestResult<object>;
/**
* A function that returns a message containing the welcome message and the
* routes of the PFun CMA Model API.
*/
export function runGet(): RunGetResult0;

type RunPostResult0 = RequestResult<object>;
/**
* A function that returns a message containing the welcome message and the
* routes of the PFun CMA Model API.
*/
export function runPost(): RunPostResult0;

type RunOptionsResult0 = RequestResult<object>;
export function runOptions(): RunOptionsResult0;

type FitPostResult0 = RequestResult<object>;
export function fitPost(): FitPostResult0;

type FitOptionsResult0 = RequestResult<object>;
export function fitOptions(): FitOptionsResult0;

type LoginSuccessGetResult0 = RequestResult<object>;
export function loginSuccessGet(): LoginSuccessGetResult0;

type LoginSuccessOptionsResult0 = RequestResult<object>;
export function loginSuccessOptions(): LoginSuccessOptionsResult0;

type LoginDexcomGetResult0 = RequestResult<object>;
export function loginDexcomGet(): LoginDexcomGetResult0;

type LoginDexcomOptionsResult0 = RequestResult<object>;
export function loginDexcomOptions(): LoginDexcomOptionsResult0;

