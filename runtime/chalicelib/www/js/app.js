var apigClient = null;
var app = {};

// Create an HTML form in your UI to input the apiKey, optional parameters, body, function selection, and method selection.
app.formCode = $(`
    <h1>API Form</h1>
  <form id="apiForm">
    <label for="apiKey">API Key:</label>
    <input type="text" id="apiKey" required><br><br>

    <label for="optionalParams">Query Parameters:</label>
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

// Retrieve the input values
var apiKey = document.getElementById('apiKey').value;
var optionalParams = document.getElementById('optionalParams').value;
var body = document.getElementById('body').value;
var selectedFunction = document.getElementById('function').value;
var selectedMethod = document.getElementById('method').value;

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
  } catch (err) {
    console.error('Failed to create api client because: ' + err);
  }
  return app;
}

function autoGrow(oField) {
  // ref: https://developer.mozilla.org/en-US/docs/Web/API/HTMLTextAreaElement
  if (oField.scrollHeight > oField.clientHeight) {
    oField.style.height = `${oField.scrollHeight}px`;
  }
}

var model_config = {
  model_config: {
    taug: 0.8
  }
};

$("#apiForm > select#function").val("routes");
$("#apiForm > select#method").val("Get");

$("#apiForm > select#function").on("change", function () {
  if (this.value == "sdk") {
    $("#apiForm > select#method").val("Get");
    $("#apiForm > select#method").prop("disabled", true);
  }
  else if (this.value == "run" && $("#apiForm > select#method").val() == "Post") {
    $("#apiForm > textarea#body").val(JSON.stringify(model_config));
  }
});

$("#apiForm > select#method").on("change", function () {
  if (this.value == "Post" && $("#apiForm > select#function").val() == "run") {
    $("#apiForm > textarea#body").val(JSON.stringify(model_config));
  }
});

$("textarea#body").on('focus', (event) => {
  setTimeout(() => {
    $(event.target).json_beautify();
    autoGrow(event.target);
  }, 5000);
  if ($(event.target).val().includes("taug")) {
    try {
      $("#taug-container").remove();
    } catch (e) {
      // pass
    }
    $("#content").append('<div id="taug-container"><label for="taug">Tau G:</label><input type="range" id="taug" min="0.01" max="5.0" step="0.01"></div>');
    $("#taug").on("input", function () {
      model_config.model_config.taug = parseFloat($("#taug").val());
      $("#apiForm > textarea#body").val(JSON.stringify(model_config));
    });
  } else {
    $("#taug-container").remove();
  }
  });
$("textarea#body").json_beautify();

var chart = null;


app.initializeApp = async () => {

  // a simple UI to input apiKey, enter any optional parameters + the body, and choose which function to call, and select the method (POST/GET). Make sure to validate the input. Handle websocket & HTTP endpoints.

  async function submitFunction(event) {
    event.preventDefault();

    // Retrieve the input values
    apiKey = document.getElementById('apiKey').value;
    optionalParams = document.getElementById('optionalParams').value;
    body = document.getElementById('body').value;
    selectedFunction = document.getElementById('function').value;
    selectedMethod = document.getElementById('method').value;

    // Validate the input values
    // ... (TODO)

    // initialize the API using the provided apiKey
    app = await initializeAPI(apiKey);

    // Handle WebSocket and HTTP endpoints based on the selected method and function
    // Call the appropriate function with the provided input values.
    try {

      console.log(selectedFunction, selectedMethod);

      var result = await apigClient[selectedFunction + selectedMethod](optionalParams, body, {
        headers: {
          Authorization: 'Bearer allow',
          Accept: '*/*'
        }
      });

      // set raw json data...
      var data = result.data;
      var string_data = JSON.stringify(data);
      console.log(string_data);
      $("#output-area").html(
        `<code>${string_data}</code>`
      );

      // update chart...
      const ctx = document.getElementById("chart");
      if (selectedFunction == 'run') {
        var arr = [];
        var Carr = [];
        const G = Object.values(data.G);
        const C = Object.values(data.c);
        Object.values(data.t).forEach((value, index) => {
          arr.push({
            x: value,
            y: G[index]
          });
          Carr.push({
            x: value,
            y: C[index]
          });
        });
        if (chart) {
          chart.clear();
          chart.destroy();
        }
        chart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: arr.map(x => x.x),
            datasets: [
              {
                label: 'Glucose',
                data: arr
              },
              {
                label: 'Cortisol',
                data: Carr
              }
            ]
          }
        });
      }

    } catch (err) {
      console.warn(`failed to access the specified endpoint: '${selectedFunction}${selectedMethod}'.\nError:`, err);
    }

  }

  // Add event listeners to handle form submission.
  document.getElementById('apiForm').addEventListener('submit', async (event) => {
    await progressIndicator(submitFunction, event);
  });
};

async function setupApp() {
  app = await app.initializeApp();

  // Get the content div element
  var contentDiv = document.getElementById("content");

  // Get the restore button element
  var restoreButton = document.createElement("button");
  restoreButton.innerText = "Toggle API Form";
  restoreButton.classList.add("btn", "btn-primary", "mt-2");
  $("ul.navbar-nav").append(`<li id="restore-button-container" class="nav-item"></li>`);
  document.getElementById("restore-button-container").appendChild(restoreButton);

  // Add a click event listener to toggle the class "minimized" on the content div when the div or restore button is clicked
  function toggleMinimized() {
    contentDiv.classList.toggle("minimized");
    if (contentDiv.classList.contains("minimized")) {
      restoreButton.innerText = "Restore API Form";
      $("#output").removeClass("col-9").addClass("col-12");
    } else {
      restoreButton.innerText = "Minimize API Form";
      $("#output").removeClass("col-12").addClass("col-9");
    }
  }

  restoreButton.addEventListener("click", toggleMinimized);
}

setupApp();