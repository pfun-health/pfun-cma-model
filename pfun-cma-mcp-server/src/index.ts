#!/usr/bin/env node
/**
 * MCP Server generated from OpenAPI spec for pfun-cma-model-backend v0.3.15
 * Generated on: 2025-09-11T18:48:28.297Z
 */

// Load environment variables from .env file
import dotenv from 'dotenv';
dotenv.config();

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
  type CallToolResult,
  type CallToolRequest
} from "@modelcontextprotocol/sdk/types.js";

import { z, ZodError } from 'zod';
import { jsonSchemaToZod } from 'json-schema-to-zod';
import axios, { type AxiosRequestConfig, type AxiosError } from 'axios';

/**
 * Type definition for JSON objects
 */
type JsonObject = Record<string, any>;

/**
 * Interface for MCP Tool Definition
 */
interface McpToolDefinition {
    name: string;
    description: string;
    inputSchema: any;
    method: string;
    pathTemplate: string;
    executionParameters: { name: string, in: string }[];
    requestBodyContentType?: string;
    securityRequirements: any[];
}

/**
 * Server configuration
 */
export const SERVER_NAME = "pfun-cma-model-backend";
export const SERVER_VERSION = "0.3.15";
export const API_BASE_URL = "";

/**
 * MCP Server instance
 */
const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    { capabilities: { tools: {} } }
);

/**
 * Map of tool definitions by name
 */
const toolDefinitionMap: Map<string, McpToolDefinition> = new Map([

  ["health_check_health_get", {
    name: "health_check_health_get",
    description: `Health check endpoint.`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/health",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["root__get", {
    name: "root__get",
    description: `Root endpoint to display the homepage.`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["params_schema_params_schema_get", {
    name: "params_schema_params_schema_get",
    description: `Params Schema`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/params/schema",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["default_params_params_default_get", {
    name: "default_params_params_default_get",
    description: `Default Params`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/params/default",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["describe_params_params_describe_post", {
    name: "describe_params_params_describe_post",
    description: `Describe a given (single) or set of parameters using CMAModelParams.describe and generate_qualitative_descriptor.
Args:
    config (Optional[BoundedCMAModelParams | Mapping]): The configuration parameters to describe.
Returns:
    dict: Dictionary of parameter descriptions and qualitative descriptors.`,
    inputSchema: {"type":"object","properties":{"requestBody":{"anyOf":[{"properties":{"t":{"anyOf":[{"type":"number"},{},{"type":"null"}],"title":"T"},"N":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"N","default":24},"d":{"type":"number","title":"D","default":0},"taup":{"type":"number","title":"Taup","default":1},"taug":{"anyOf":[{"type":"number"},{}],"title":"Taug","default":1},"B":{"type":"number","title":"B","default":0.05},"Cm":{"type":"number","title":"Cm","default":0},"toff":{"type":"number","title":"Toff","default":0},"tM":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"number"}],"title":"Tm","default":[7,11,17.5]},"seed":{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"null"}],"title":"Seed"},"eps":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Eps","default":1e-18}},"type":"object","title":"CMAModelParams","description":"Represents the parameters for a CMA model.\n\nArgs:\n    t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.\n    N (int, optional): Number of time points. Defaults to 24.\n    d (float, optional): Time zone offset (hours). Defaults to 0.0.\n    taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.\n    taug (float, optional): Glucose response time constant. Defaults to 1.0.\n    B (float, optional): Glucose Bias constant. Defaults to 0.05.\n    Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.\n    toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.\n    tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).\n    seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.\n    eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18."},{"type":"object"}],"title":"Params","description":"The JSON request body."}},"required":["requestBody"]},
    method: "post",
    pathTemplate: "/params/describe",
    executionParameters: [],
    requestBodyContentType: "application/json",
    securityRequirements: []
  }],
  ["tabulate_params_params_tabulate_post", {
    name: "tabulate_params_params_tabulate_post",
    description: `Generate a markdown table of a given (single) or set of parameters using CMAModelParams.generate_markdown_table.
Args:
    config (Optional[BoundedCMAModelParams | Mapping]): The configuration parameters to describe.
Returns:
    dict: Dictionary of parameter descriptions and qualitative descriptors.`,
    inputSchema: {"type":"object","properties":{"requestBody":{"anyOf":[{"properties":{"t":{"anyOf":[{"type":"number"},{},{"type":"null"}],"title":"T"},"N":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"N","default":24},"d":{"type":"number","title":"D","default":0},"taup":{"type":"number","title":"Taup","default":1},"taug":{"anyOf":[{"type":"number"},{}],"title":"Taug","default":1},"B":{"type":"number","title":"B","default":0.05},"Cm":{"type":"number","title":"Cm","default":0},"toff":{"type":"number","title":"Toff","default":0},"tM":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"number"}],"title":"Tm","default":[7,11,17.5]},"seed":{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"null"}],"title":"Seed"},"eps":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Eps","default":1e-18}},"type":"object","title":"CMAModelParams","description":"Represents the parameters for a CMA model.\n\nArgs:\n    t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.\n    N (int, optional): Number of time points. Defaults to 24.\n    d (float, optional): Time zone offset (hours). Defaults to 0.0.\n    taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.\n    taug (float, optional): Glucose response time constant. Defaults to 1.0.\n    B (float, optional): Glucose Bias constant. Defaults to 0.05.\n    Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.\n    toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.\n    tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).\n    seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.\n    eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18."},{"type":"object"}],"title":"Params","description":"The JSON request body."}},"required":["requestBody"]},
    method: "post",
    pathTemplate: "/params/tabulate",
    executionParameters: [],
    requestBodyContentType: "application/json",
    securityRequirements: []
  }],
  ["get_sample_dataset_data_sample_download_get", {
    name: "get_sample_dataset_data_sample_download_get",
    description: `(slow) Download the sample dataset with optional row limit.

Args:
    request (Request): The FastAPI request object.
    nrows (int): The number of rows to return. If -1, return the full dataset.`,
    inputSchema: {"type":"object","properties":{"nrows":{"type":"number","default":23,"title":"Nrows"}}},
    method: "get",
    pathTemplate: "/data/sample/download",
    executionParameters: [{"name":"nrows","in":"query"}],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["stream_sample_dataset_data_sample_stream_get", {
    name: "stream_sample_dataset_data_sample_stream_get",
    description: `(faster stream) Stream the sample dataset with optional row limit.
Args:
    request (Request): The FastAPI request object.
    nrows (int): The number of rows to return. If -1, return the full dataset.`,
    inputSchema: {"type":"object","properties":{"nrows":{"type":"number","default":23,"title":"Nrows"}}},
    method: "get",
    pathTemplate: "/data/sample/stream",
    executionParameters: [{"name":"nrows","in":"query"}],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["translate_model_results_by_language_translate_results_post", {
    name: "translate_model_results_by_language_translate_results_post",
    description: `Translate model results between Python and JavaScript formats.`,
    inputSchema: {"type":"object","properties":{"from_lang":{"enum":["python","javascript"],"type":"string","title":"From Lang"},"requestBody":{"type":"object","title":"Results","description":"The JSON request body."}},"required":["from_lang","requestBody"]},
    method: "post",
    pathTemplate: "/translate-results",
    executionParameters: [{"name":"from_lang","in":"query"}],
    requestBodyContentType: "application/json",
    securityRequirements: []
  }],
  ["run_model_run_post", {
    name: "run_model_run_post",
    description: `Runs the CMA model.`,
    inputSchema: {"type":"object","properties":{"config":{"anyOf":[{"properties":{"t":{"anyOf":[{"type":"number"},{},{"type":"null"}],"title":"T"},"N":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"N","default":24},"d":{"type":"number","title":"D","default":0},"taup":{"type":"number","title":"Taup","default":1},"taug":{"anyOf":[{"type":"number"},{}],"title":"Taug","default":1},"B":{"type":"number","title":"B","default":0.05},"Cm":{"type":"number","title":"Cm","default":0},"toff":{"type":"number","title":"Toff","default":0},"tM":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"number"}],"title":"Tm","default":[7,11,17.5]},"seed":{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"null"}],"title":"Seed"},"eps":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Eps","default":1e-18}},"type":"object","title":"CMAModelParams","description":"Represents the parameters for a CMA model.\n\nArgs:\n    t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.\n    N (int, optional): Number of time points. Defaults to 24.\n    d (float, optional): Time zone offset (hours). Defaults to 0.0.\n    taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.\n    taug (float, optional): Glucose response time constant. Defaults to 1.0.\n    B (float, optional): Glucose Bias constant. Defaults to 0.05.\n    Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.\n    toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.\n    tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).\n    seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.\n    eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18."},{"type":"null"}],"title":"Config"}}},
    method: "post",
    pathTemplate: "/run",
    executionParameters: [{"name":"config","in":"query"}],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["run_at_time_route_run_at_time_post", {
    name: "run_at_time_route_run_at_time_post",
    description: `Run the CMA model at a specific time.

Parameters:
- t0 (float | int): The start time (in decimal hours).
- t1 (float | int): The end time (in decimal hours).
- n (int): The number of samples.
- config (CMAModelParams): The model configuration parameters.`,
    inputSchema: {"type":"object","properties":{"t0":{"anyOf":[{"type":"number"},{"type":"integer"}],"title":"T0"},"t1":{"anyOf":[{"type":"number"},{"type":"integer"}],"title":"T1"},"n":{"type":"number","title":"N"},"requestBody":{"anyOf":[{"properties":{"t":{"anyOf":[{"type":"number"},{},{"type":"null"}],"title":"T"},"N":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"N","default":24},"d":{"type":"number","title":"D","default":0},"taup":{"type":"number","title":"Taup","default":1},"taug":{"anyOf":[{"type":"number"},{}],"title":"Taug","default":1},"B":{"type":"number","title":"B","default":0.05},"Cm":{"type":"number","title":"Cm","default":0},"toff":{"type":"number","title":"Toff","default":0},"tM":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"number"}],"title":"Tm","default":[7,11,17.5]},"seed":{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"null"}],"title":"Seed"},"eps":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Eps","default":1e-18}},"type":"object","title":"CMAModelParams","description":"Represents the parameters for a CMA model.\n\nArgs:\n    t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.\n    N (int, optional): Number of time points. Defaults to 24.\n    d (float, optional): Time zone offset (hours). Defaults to 0.0.\n    taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.\n    taug (float, optional): Glucose response time constant. Defaults to 1.0.\n    B (float, optional): Glucose Bias constant. Defaults to 0.05.\n    Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.\n    toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.\n    tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).\n    seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.\n    eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18."},{"type":"null"}],"title":"Config","description":"The JSON request body."}},"required":["t0","t1","n"]},
    method: "post",
    pathTemplate: "/run-at-time",
    executionParameters: [{"name":"t0","in":"query"},{"name":"t1","in":"query"},{"name":"n","in":"query"}],
    requestBodyContentType: "application/json",
    securityRequirements: []
  }],
  ["health_check_run_at_time_health_ws_run_at_time_get", {
    name: "health_check_run_at_time_health_ws_run_at_time_get",
    description: `Health check endpoint for the 'run-at-time' WebSocket functionality.`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/health/ws/run-at-time",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["demo_run_at_time_demo_run_at_time_get", {
    name: "demo_run_at_time_demo_run_at_time_get",
    description: `Demo UI endpoint to run the model at a specific time (using websockets).`,
    inputSchema: {"type":"object","properties":{}},
    method: "get",
    pathTemplate: "/demo/run-at-time",
    executionParameters: [],
    requestBodyContentType: undefined,
    securityRequirements: []
  }],
  ["fit_model_to_data_fit_post", {
    name: "fit_model_to_data_fit_post",
    description: `Fit Model To Data`,
    inputSchema: {"type":"object","properties":{"requestBody":{"properties":{"data":{"anyOf":[{"type":"object"},{"type":"string"}],"title":"Data"},"config":{"anyOf":[{"properties":{"t":{"anyOf":[{"type":"number"},{},{"type":"null"}],"title":"T"},"N":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"N","default":24},"d":{"type":"number","title":"D","default":0},"taup":{"type":"number","title":"Taup","default":1},"taug":{"anyOf":[{"type":"number"},{}],"title":"Taug","default":1},"B":{"type":"number","title":"B","default":0.05},"Cm":{"type":"number","title":"Cm","default":0},"toff":{"type":"number","title":"Toff","default":0},"tM":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"number"}],"title":"Tm","default":[7,11,17.5]},"seed":{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"null"}],"title":"Seed"},"eps":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Eps","default":1e-18}},"type":"object","title":"CMAModelParams","description":"Represents the parameters for a CMA model.\n\nArgs:\n    t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.\n    N (int, optional): Number of time points. Defaults to 24.\n    d (float, optional): Time zone offset (hours). Defaults to 0.0.\n    taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.\n    taug (float, optional): Glucose response time constant. Defaults to 1.0.\n    B (float, optional): Glucose Bias constant. Defaults to 0.05.\n    Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.\n    toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.\n    tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).\n    seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.\n    eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18."},{"type":"string"},{"type":"null"}],"title":"Config"}},"type":"object","required":["data"],"title":"Body_fit_model_to_data_fit_post","description":"The JSON request body."}},"required":["requestBody"]},
    method: "post",
    pathTemplate: "/fit",
    executionParameters: [],
    requestBodyContentType: "application/json",
    securityRequirements: []
  }],
]);

/**
 * Security schemes from the OpenAPI spec
 */
const securitySchemes =   {};


server.setRequestHandler(ListToolsRequestSchema, async () => {
  const toolsForClient: Tool[] = Array.from(toolDefinitionMap.values()).map(def => ({
    name: def.name,
    description: def.description,
    inputSchema: def.inputSchema
  }));
  return { tools: toolsForClient };
});


server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
  const { name: toolName, arguments: toolArgs } = request.params;
  const toolDefinition = toolDefinitionMap.get(toolName);
  if (!toolDefinition) {
    console.error(`Error: Unknown tool requested: ${toolName}`);
    return { content: [{ type: "text", text: `Error: Unknown tool requested: ${toolName}` }] };
  }
  return await executeApiTool(toolName, toolDefinition, toolArgs ?? {}, securitySchemes);
});



/**
 * Type definition for cached OAuth tokens
 */
interface TokenCacheEntry {
    token: string;
    expiresAt: number;
}

/**
 * Declare global __oauthTokenCache property for TypeScript
 */
declare global {
    var __oauthTokenCache: Record<string, TokenCacheEntry> | undefined;
}

/**
 * Acquires an OAuth2 token using client credentials flow
 * 
 * @param schemeName Name of the security scheme
 * @param scheme OAuth2 security scheme
 * @returns Acquired token or null if unable to acquire
 */
async function acquireOAuth2Token(schemeName: string, scheme: any): Promise<string | null | undefined> {
    try {
        // Check if we have the necessary credentials
        const clientId = process.env[`OAUTH_CLIENT_ID_SCHEMENAME`];
        const clientSecret = process.env[`OAUTH_CLIENT_SECRET_SCHEMENAME`];
        const scopes = process.env[`OAUTH_SCOPES_SCHEMENAME`];
        
        if (!clientId || !clientSecret) {
            console.error(`Missing client credentials for OAuth2 scheme '${schemeName}'`);
            return null;
        }
        
        // Initialize token cache if needed
        if (typeof global.__oauthTokenCache === 'undefined') {
            global.__oauthTokenCache = {};
        }
        
        // Check if we have a cached token
        const cacheKey = `${schemeName}_${clientId}`;
        const cachedToken = global.__oauthTokenCache[cacheKey];
        const now = Date.now();
        
        if (cachedToken && cachedToken.expiresAt > now) {
            console.error(`Using cached OAuth2 token for '${schemeName}' (expires in ${Math.floor((cachedToken.expiresAt - now) / 1000)} seconds)`);
            return cachedToken.token;
        }
        
        // Determine token URL based on flow type
        let tokenUrl = '';
        if (scheme.flows?.clientCredentials?.tokenUrl) {
            tokenUrl = scheme.flows.clientCredentials.tokenUrl;
            console.error(`Using client credentials flow for '${schemeName}'`);
        } else if (scheme.flows?.password?.tokenUrl) {
            tokenUrl = scheme.flows.password.tokenUrl;
            console.error(`Using password flow for '${schemeName}'`);
        } else {
            console.error(`No supported OAuth2 flow found for '${schemeName}'`);
            return null;
        }
        
        // Prepare the token request
        let formData = new URLSearchParams();
        formData.append('grant_type', 'client_credentials');
        
        // Add scopes if specified
        if (scopes) {
            formData.append('scope', scopes);
        }
        
        console.error(`Requesting OAuth2 token from ${tokenUrl}`);
        
        // Make the token request
        const response = await axios({
            method: 'POST',
            url: tokenUrl,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`
            },
            data: formData.toString()
        });
        
        // Process the response
        if (response.data?.access_token) {
            const token = response.data.access_token;
            const expiresIn = response.data.expires_in || 3600; // Default to 1 hour
            
            // Cache the token
            global.__oauthTokenCache[cacheKey] = {
                token,
                expiresAt: now + (expiresIn * 1000) - 60000 // Expire 1 minute early
            };
            
            console.error(`Successfully acquired OAuth2 token for '${schemeName}' (expires in ${expiresIn} seconds)`);
            return token;
        } else {
            console.error(`Failed to acquire OAuth2 token for '${schemeName}': No access_token in response`);
            return null;
        }
    } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error(`Error acquiring OAuth2 token for '${schemeName}':`, errorMessage);
        return null;
    }
}


/**
 * Executes an API tool with the provided arguments
 * 
 * @param toolName Name of the tool to execute
 * @param definition Tool definition
 * @param toolArgs Arguments provided by the user
 * @param allSecuritySchemes Security schemes from the OpenAPI spec
 * @returns Call tool result
 */
async function executeApiTool(
    toolName: string,
    definition: McpToolDefinition,
    toolArgs: JsonObject,
    allSecuritySchemes: Record<string, any>
): Promise<CallToolResult> {
  try {
    // Validate arguments against the input schema
    let validatedArgs: JsonObject;
    try {
        const zodSchema = getZodSchemaFromJsonSchema(definition.inputSchema, toolName);
        const argsToParse = (typeof toolArgs === 'object' && toolArgs !== null) ? toolArgs : {};
        validatedArgs = zodSchema.parse(argsToParse);
    } catch (error: unknown) {
        if (error instanceof ZodError) {
            const validationErrorMessage = `Invalid arguments for tool '${toolName}': ${error.errors.map(e => `${e.path.join('.')} (${e.code}): ${e.message}`).join(', ')}`;
            return { content: [{ type: 'text', text: validationErrorMessage }] };
        } else {
             const errorMessage = error instanceof Error ? error.message : String(error);
             return { content: [{ type: 'text', text: `Internal error during validation setup: ${errorMessage}` }] };
        }
    }

    // Prepare URL, query parameters, headers, and request body
    let urlPath = definition.pathTemplate;
    const queryParams: Record<string, any> = {};
    const headers: Record<string, string> = { 'Accept': 'application/json' };
    let requestBodyData: any = undefined;

    // Apply parameters to the URL path, query, or headers
    definition.executionParameters.forEach((param) => {
        const value = validatedArgs[param.name];
        if (typeof value !== 'undefined' && value !== null) {
            if (param.in === 'path') {
                urlPath = urlPath.replace(`{${param.name}}`, encodeURIComponent(String(value)));
            }
            else if (param.in === 'query') {
                queryParams[param.name] = value;
            }
            else if (param.in === 'header') {
                headers[param.name.toLowerCase()] = String(value);
            }
        }
    });

    // Ensure all path parameters are resolved
    if (urlPath.includes('{')) {
        throw new Error(`Failed to resolve path parameters: ${urlPath}`);
    }
    
    // Construct the full URL
    const requestUrl = API_BASE_URL ? `${API_BASE_URL}${urlPath}` : urlPath;

    // Handle request body if needed
    if (definition.requestBodyContentType && typeof validatedArgs['requestBody'] !== 'undefined') {
        requestBodyData = validatedArgs['requestBody'];
        headers['content-type'] = definition.requestBodyContentType;
    }


    // Apply security requirements if available
    // Security requirements use OR between array items and AND within each object
    const appliedSecurity = definition.securityRequirements?.find(req => {
        // Try each security requirement (combined with OR)
        return Object.entries(req).every(([schemeName, scopesArray]) => {
            const scheme = allSecuritySchemes[schemeName];
            if (!scheme) return false;
            
            // API Key security (header, query, cookie)
            if (scheme.type === 'apiKey') {
                return !!process.env[`API_KEY_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
            }
            
            // HTTP security (basic, bearer)
            if (scheme.type === 'http') {
                if (scheme.scheme?.toLowerCase() === 'bearer') {
                    return !!process.env[`BEARER_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                }
                else if (scheme.scheme?.toLowerCase() === 'basic') {
                    return !!process.env[`BASIC_USERNAME_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`] && 
                           !!process.env[`BASIC_PASSWORD_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                }
            }
            
            // OAuth2 security
            if (scheme.type === 'oauth2') {
                // Check for pre-existing token
                if (process.env[`OAUTH_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`]) {
                    return true;
                }
                
                // Check for client credentials for auto-acquisition
                if (process.env[`OAUTH_CLIENT_ID_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`] &&
                    process.env[`OAUTH_CLIENT_SECRET_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`]) {
                    // Verify we have a supported flow
                    if (scheme.flows?.clientCredentials || scheme.flows?.password) {
                        return true;
                    }
                }
                
                return false;
            }
            
            // OpenID Connect
            if (scheme.type === 'openIdConnect') {
                return !!process.env[`OPENID_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
            }
            
            return false;
        });
    });

    // If we found matching security scheme(s), apply them
    if (appliedSecurity) {
        // Apply each security scheme from this requirement (combined with AND)
        for (const [schemeName, scopesArray] of Object.entries(appliedSecurity)) {
            const scheme = allSecuritySchemes[schemeName];
            
            // API Key security
            if (scheme?.type === 'apiKey') {
                const apiKey = process.env[`API_KEY_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                if (apiKey) {
                    if (scheme.in === 'header') {
                        headers[scheme.name.toLowerCase()] = apiKey;
                        console.error(`Applied API key '${schemeName}' in header '${scheme.name}'`);
                    }
                    else if (scheme.in === 'query') {
                        queryParams[scheme.name] = apiKey;
                        console.error(`Applied API key '${schemeName}' in query parameter '${scheme.name}'`);
                    }
                    else if (scheme.in === 'cookie') {
                        // Add the cookie, preserving other cookies if they exist
                        headers['cookie'] = `${scheme.name}=${apiKey}${headers['cookie'] ? `; ${headers['cookie']}` : ''}`;
                        console.error(`Applied API key '${schemeName}' in cookie '${scheme.name}'`);
                    }
                }
            } 
            // HTTP security (Bearer or Basic)
            else if (scheme?.type === 'http') {
                if (scheme.scheme?.toLowerCase() === 'bearer') {
                    const token = process.env[`BEARER_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    if (token) {
                        headers['authorization'] = `Bearer ${token}`;
                        console.error(`Applied Bearer token for '${schemeName}'`);
                    }
                } 
                else if (scheme.scheme?.toLowerCase() === 'basic') {
                    const username = process.env[`BASIC_USERNAME_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    const password = process.env[`BASIC_PASSWORD_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    if (username && password) {
                        headers['authorization'] = `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`;
                        console.error(`Applied Basic authentication for '${schemeName}'`);
                    }
                }
            }
            // OAuth2 security
            else if (scheme?.type === 'oauth2') {
                // First try to use a pre-provided token
                let token = process.env[`OAUTH_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                
                // If no token but we have client credentials, try to acquire a token
                if (!token && (scheme.flows?.clientCredentials || scheme.flows?.password)) {
                    console.error(`Attempting to acquire OAuth token for '${schemeName}'`);
                    token = (await acquireOAuth2Token(schemeName, scheme)) ?? '';
                }
                
                // Apply token if available
                if (token) {
                    headers['authorization'] = `Bearer ${token}`;
                    console.error(`Applied OAuth2 token for '${schemeName}'`);
                    
                    // List the scopes that were requested, if any
                    const scopes = scopesArray as string[];
                    if (scopes && scopes.length > 0) {
                        console.error(`Requested scopes: ${scopes.join(', ')}`);
                    }
                }
            }
            // OpenID Connect
            else if (scheme?.type === 'openIdConnect') {
                const token = process.env[`OPENID_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                if (token) {
                    headers['authorization'] = `Bearer ${token}`;
                    console.error(`Applied OpenID Connect token for '${schemeName}'`);
                    
                    // List the scopes that were requested, if any
                    const scopes = scopesArray as string[];
                    if (scopes && scopes.length > 0) {
                        console.error(`Requested scopes: ${scopes.join(', ')}`);
                    }
                }
            }
        }
    } 
    // Log warning if security is required but not available
    else if (definition.securityRequirements?.length > 0) {
        // First generate a more readable representation of the security requirements
        const securityRequirementsString = definition.securityRequirements
            .map(req => {
                const parts = Object.entries(req)
                    .map(([name, scopesArray]) => {
                        const scopes = scopesArray as string[];
                        if (scopes.length === 0) return name;
                        return `${name} (scopes: ${scopes.join(', ')})`;
                    })
                    .join(' AND ');
                return `[${parts}]`;
            })
            .join(' OR ');
            
        console.warn(`Tool '${toolName}' requires security: ${securityRequirementsString}, but no suitable credentials found.`);
    }
    

    // Prepare the axios request configuration
    const config: AxiosRequestConfig = {
      method: definition.method.toUpperCase(), 
      url: requestUrl, 
      params: queryParams, 
      headers: headers,
      ...(requestBodyData !== undefined && { data: requestBodyData }),
    };

    // Log request info to stderr (doesn't affect MCP output)
    console.error(`Executing tool "${toolName}": ${config.method} ${config.url}`);
    
    // Execute the request
    const response = await axios(config);

    // Process and format the response
    let responseText = '';
    const contentType = response.headers['content-type']?.toLowerCase() || '';
    
    // Handle JSON responses
    if (contentType.includes('application/json') && typeof response.data === 'object' && response.data !== null) {
         try { 
             responseText = JSON.stringify(response.data, null, 2); 
         } catch (e) { 
             responseText = "[Stringify Error]"; 
         }
    } 
    // Handle string responses
    else if (typeof response.data === 'string') { 
         responseText = response.data; 
    }
    // Handle other response types
    else if (response.data !== undefined && response.data !== null) { 
         responseText = String(response.data); 
    }
    // Handle empty responses
    else { 
         responseText = `(Status: ${response.status} - No body content)`; 
    }
    
    // Return formatted response
    return { 
        content: [ 
            { 
                type: "text", 
                text: `API Response (Status: ${response.status}):\n${responseText}` 
            } 
        ], 
    };

  } catch (error: unknown) {
    // Handle errors during execution
    let errorMessage: string;
    
    // Format Axios errors specially
    if (axios.isAxiosError(error)) { 
        errorMessage = formatApiError(error); 
    }
    // Handle standard errors
    else if (error instanceof Error) { 
        errorMessage = error.message; 
    }
    // Handle unexpected error types
    else { 
        errorMessage = 'Unexpected error: ' + String(error); 
    }
    
    // Log error to stderr
    console.error(`Error during execution of tool '${toolName}':`, errorMessage);
    
    // Return error message to client
    return { content: [{ type: "text", text: errorMessage }] };
  }
}


/**
 * Main function to start the server
 */
async function main() {
// Set up stdio transport
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(`${SERVER_NAME} MCP Server (v${SERVER_VERSION}) running on stdio${API_BASE_URL ? `, proxying API at ${API_BASE_URL}` : ''}`);
  } catch (error) {
    console.error("Error during server startup:", error);
    process.exit(1);
  }
}

/**
 * Cleanup function for graceful shutdown
 */
async function cleanup() {
    console.error("Shutting down MCP server...");
    process.exit(0);
}

// Register signal handlers
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Start the server
main().catch((error) => {
  console.error("Fatal error in main execution:", error);
  process.exit(1);
});

/**
 * Formats API errors for better readability
 * 
 * @param error Axios error
 * @returns Formatted error message
 */
function formatApiError(error: AxiosError): string {
    let message = 'API request failed.';
    if (error.response) {
        message = `API Error: Status ${error.response.status} (${error.response.statusText || 'Status text not available'}). `;
        const responseData = error.response.data;
        const MAX_LEN = 200;
        if (typeof responseData === 'string') { 
            message += `Response: ${responseData.substring(0, MAX_LEN)}${responseData.length > MAX_LEN ? '...' : ''}`; 
        }
        else if (responseData) { 
            try { 
                const jsonString = JSON.stringify(responseData); 
                message += `Response: ${jsonString.substring(0, MAX_LEN)}${jsonString.length > MAX_LEN ? '...' : ''}`; 
            } catch { 
                message += 'Response: [Could not serialize data]'; 
            } 
        }
        else { 
            message += 'No response body received.'; 
        }
    } else if (error.request) {
        message = 'API Network Error: No response received from server.';
        if (error.code) message += ` (Code: ${error.code})`;
    } else { 
        message += `API Request Setup Error: ${error.message}`; 
    }
    return message;
}

/**
 * Converts a JSON Schema to a Zod schema for runtime validation
 * 
 * @param jsonSchema JSON Schema
 * @param toolName Tool name for error reporting
 * @returns Zod schema
 */
function getZodSchemaFromJsonSchema(jsonSchema: any, toolName: string): z.ZodTypeAny {
    if (typeof jsonSchema !== 'object' || jsonSchema === null) { 
        return z.object({}).passthrough(); 
    }
    try {
        const zodSchemaString = jsonSchemaToZod(jsonSchema);
        const zodSchema = eval(zodSchemaString);
        if (typeof zodSchema?.parse !== 'function') { 
            throw new Error('Eval did not produce a valid Zod schema.'); 
        }
        return zodSchema as z.ZodTypeAny;
    } catch (err: any) {
        console.error(`Failed to generate/evaluate Zod schema for '${toolName}':`, err);
        return z.object({}).passthrough();
    }
}
