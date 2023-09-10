
var apigClient = null;
var app = {};

$.fn.json_beautify = function () {
  // ref: https://stackoverflow.com/a/62060316/1871569
  try {
    this.each(function () {
      var el = $(this),
        obj = JSON.parse(el.val()),
        pretty = JSON.stringify(obj, undefined, 4);
      el.val(pretty);
    });
  } catch (err) {
    console.warn('Failed to beautify JSON because: ' + err);
  }
};

async function initializeAPI(apiKey) {
  if (apigClient !== null) {
    return app;
  }
  try {
    apigClient = await apigClientFactory.newClient({
      apiKey: apiKey
    });
    console.log('...initialized API.');
    app.apigClient = apigClient;
  } catch (err) {
    console.error('Failed to create api client because: ' + err);
    return app;
  }
}

function autoGrow(oField) {
  // ref: https://developer.mozilla.org/en-US/docs/Web/API/HTMLTextAreaElement
  if (oField.scrollHeight > oField.clientHeight) {
    oField.style.height = `${oField.scrollHeight}px`;
  }
}


app.initializeApp = async () => {

  // a simple UI to input apiKey, enter any optional parameters + the body, and choose which function to call, and select the method (POST/GET). Make sure to validate the input. Handle websocket & HTTP endpoints.

  // Create an HTML form in your UI to input the apiKey, optional parameters, body, function selection, and method selection.
  app.formCode = $(`
    <h1>API Form</h1>
  <form id="apiForm">
    <label for="apiKey">API Key:</label>
    <input type="text" id="apiKey" required><br><br>

    <label for="optionalParams">Query Parameters:</label>
    <input type="text" id="optionalParams"><br><br>

    <label for="body">Body:</label>
    <textarea id="body">{}</textarea><br><br>

    <label for="function">Function:</label>
    <select id="function">
      <option value="sdk">SDK</option>
      <option value="log">Log</option>
      <option value="run">Run</option>
      <option value="root">Root</option>
      <option value="runOptions">Run Options</option>
      <option value="fit">Fit</option>
      <option value="routes">Routes</option>
      <option value="runAtTime">Run At Time</option>
    </select><br><br>

    <label for="method">Method:</label>
    <select id="method">
      <option value="Post">POST</option>
      <option value="Get">GET</option>
    </select><br><br>

    <button type="submit">Submit</button>
  </form>
    `);
  $('#content').html(''); // clear the content of the page
  $('#content').append(app.formCode);
  $("#apiForm > select#function").val("run");
  $("#apiForm > select#function").on("change", function () {
    if (this.value == "sdk") {
      $("#apiForm > select#method").val("Get");
      $("#apiForm > select#method").prop("disabled", true);
    }
  });
  $("textarea#body").on('input', (event) => {
    setTimeout(() => {
      $(event.target).json_beautify();
      autoGrow(event.target);
    }, 1000);
  });
  $("textarea#body").json_beautify();

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
    app = await initializeAPI(apiKey);

    // Handle WebSocket and HTTP endpoints based on the selected method and function
    // Call the appropriate function with the provided input values.
    try {
      console.log(selectedFunction, selectedMethod);
      apigClient[selectedFunction + selectedMethod](optionalParams, body, {
        headers: {
          Authorization: 'Bearer allow'
        }
      })
        .then(async (response) => {
          return await response.json();
        })
        .then((json_data) => {
          $("#output-area").html(
            `<pre><code>${json_data}</code></pre>`
          );
        })
        .catch((err) => {
          console.warn(`failed to access the specified endpoint: '${selectedFunction}${selectedMethod}'.\nError:`, err);
        })
    } catch (err) {
      console.warn(`failed to access the specified endpoint: '${selectedFunction}${selectedMethod}'.\nError:`, err);
    }

  });
};

document.addEventListener('DOMContentLoaded', async function () {
  app = await app.initializeApp();
});