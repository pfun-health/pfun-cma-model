
var apigClient = null;
var app = {};

async function initializeAPI(apiKey) {
    try {
        apigClient = apigClientFactory.newClient({
            apiKey: apiKey
        });
        console.log('API initialized');
    } catch (err) {
        console.error('Failed to create api client because: ' + err);
        return;
    }
}

app.initializeApp = async () => {

    // TODO: write a simple UI to input apiKey, enter any optional parameters + the body, and choose which function to call, and select the method (POST/GET). Make sure to validate the input. Handle websocket & HTTP endpoints.

    // Create an HTML form in your UI to input the apiKey, optional parameters, body, function selection, and method selection.
    app.formCode = $(`
    <h1>API Form</h1>
  <form id="apiForm">
    <label for="apiKey">API Key:</label>
    <input type="text" id="apiKey" required><br><br>

    <label for="optionalParams">Optional Parameters:</label>
    <input type="text" id="optionalParams"><br><br>

    <label for="body">Body:</label>
    <textarea id="body"></textarea><br><br>

    <label for="function">Function:</label>
    <select id="function">
      <option value="sdk">SDK</option>
      <option value="log">Log</option>
      <option value="run">Run</option>
      <option value="root">Root</option>
      <option value="runOptions">Run Options</option>
      <option value="fit">Fit</option>
      <option value="routes">Routes</option>
    </select><br><br>

    <label for="method">Method:</label>
    <select id="method">
      <option value="Post">POST</option>
      <option value="Get">GET</option>
    </select><br><br>

    <button type="submit">Submit</button>
  </form>
    `);
  $('#content').append(app.formCode);
  $("#apiForm > select#function").val("Run");
  $("#apiForm > select#function").on("change", function () {
    if (this.value == "sdk") {
      $("#apiForm > select#method").val("Get");
      $("#apiForm > select#method").prop("disabled", true);
    }
  });

    // Add event listeners to handle form submission.
    document.getElementById('apiForm').addEventListener('submit', async function (event) {
        event.preventDefault();

        // Retrieve the input values
        var apiKey = document.getElementById('apiKey').value;
        var optionalParams = document.getElementById('optionalParams').value;
        var body = document.getElementById('body').value;
        var selectedFunction = document.getElementById('function').value;
        var selectedMethod = document.getElementById('method').value;

        // Validate the input values
      // ... (TODO)

        // initialize the API using the provided apiKey
        await initializeAPI(apiKey);

      app = Object.assign(app, {
        'sdkGet': apigClient.sdkGet,
        'logGet': apigClient.logGet,
        'runGet': apigClient.runGet,
        'rootGet': apigClient.rootGet,
        'runPost': apigClient.runPost,
        'logPost': apigClient.logPost,
        'fitPost': apigClient.fitPost,
        'routesGet': apigClient.routesGet,
        'runOptions': apigClient.runOptions
      });

        // Handle WebSocket and HTTP endpoints based on the selected method and function
        // Call the appropriate function with the provided input values.
        app[selectedFunction + selectedMethod](optionalParams, body);

    });
};

document.addEventListener('DOMContentLoaded', async function () {
  await app.initializeApp();
});