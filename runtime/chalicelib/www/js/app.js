// Initialize global variables

var apigClient = null;
var app = {};

// input values
var apiKey = null;
var optionalParams = null;
var body = null;
var selectedFunction = null;
var selectedMethod = null;

// Create an HTML form in your UI to input the apiKey, optional parameters, body, function selection, and method selection.
// Function to generate the API form based on the JSON response
async function generateApiForm() {

  // Example JSON response from /routes endpoint
  const routesData = await axios.get(window.location.href + '/routes', {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
  }).then((response) => {
    return response.data;
  }).catch((error) => {
    console.error('Failed to get routes data because: ', error);
  });

  const formContainer = document.getElementById('apiForm');

  // Create a label for API Key
  const apiKeyLabel = document.createElement('label');
  apiKeyLabel.setAttribute('for', 'apiKey');
  apiKeyLabel.textContent = 'API Key:';
  formContainer.appendChild(apiKeyLabel);

  // Create an input field for API Key
  const apiKeyInput = document.createElement('input');
  apiKeyInput.setAttribute('type', 'password');
  apiKeyInput.setAttribute('id', 'apiKey');
  apiKeyInput.setAttribute('required', 'true');
  formContainer.appendChild(apiKeyInput);

  // Create a line break
  formContainer.appendChild(document.createElement('br'));

  // Create a label for Query Parameters
  const queryParamsLabel = document.createElement('label');
  queryParamsLabel.setAttribute('for', 'optionalParams');
  queryParamsLabel.textContent = 'Query Parameters:';
  formContainer.appendChild(queryParamsLabel);

  // Create an input field for Query Parameters
  const queryParamsInput = document.createElement('input');
  queryParamsInput.setAttribute('type', 'text');
  queryParamsInput.setAttribute('id', 'optionalParams');
  formContainer.appendChild(queryParamsInput);

  // Create a line break
  formContainer.appendChild(document.createElement('br'));

  // Create a label for Body
  const bodyLabel = document.createElement('label');
  bodyLabel.setAttribute('for', 'body');
  bodyLabel.textContent = 'Body:';
  formContainer.appendChild(bodyLabel);

  // Create a textarea for the Body
  const bodyTextarea = document.createElement('textarea');
  bodyTextarea.setAttribute('id', 'body');
  formContainer.appendChild(bodyTextarea);

  // Create a line break
  formContainer.appendChild(document.createElement('br'));

  // Create a label for Function
  const functionLabel = document.createElement('label');
  functionLabel.setAttribute('for', 'function');
  functionLabel.textContent = 'Function:';
  formContainer.appendChild(functionLabel);

  // Create a select dropdown for Function
  const functionSelect = document.createElement('select');
  functionSelect.setAttribute('id', 'function');

  // Add options for Function based on the routesData
  for (const route in routesData) {
    const option = document.createElement('option');
    option.setAttribute('value', route);
    option.textContent = route;
    functionSelect.appendChild(option);
  }

  formContainer.appendChild(functionSelect);

  // Create a line break
  formContainer.appendChild(document.createElement('br'));

  // Create a label for Method
  const methodLabel = document.createElement('label');
  methodLabel.setAttribute('for', 'method');
  methodLabel.textContent = 'Method:';
  formContainer.appendChild(methodLabel);

  // Create a select dropdown for Method
  const methodSelect = document.createElement('select');
  methodSelect.setAttribute('id', 'method');

  // Add options for Method (POST, GET, OPTIONS)
  const methods = ['POST', 'GET', 'OPTIONS'];
  for (const method of methods) {
    const option = document.createElement('option');
    option.setAttribute('value', method);
    option.textContent = method;
    methodSelect.appendChild(option);
  }

  formContainer.appendChild(methodSelect);

  // Create a line break
  formContainer.appendChild(document.createElement('br'));

  // Create a Submit button
  const submitButton = document.createElement('button');
  submitButton.setAttribute('type', 'submit');
  submitButton.textContent = 'Submit';
  formContainer.appendChild(submitButton);

  if (localStorage.getItem('PFUN_CMA_API_KEY')) {
    apiKeyInput.value = localStorage.getItem('PFUN_CMA_API_KEY');
  }


  $("#apiForm > select#function").val("/routes");
  $("#apiForm > select#method").val("GET");

  $("#apiForm > select#function").on("change", function () {
    if (this.value == "run" && $("#apiForm > select#method").val() == "Post") {
      $("#apiForm > textarea#body").val(JSON.stringify(model_config));
    }
  });

  $("#apiForm > select#method").on("change", function () {
    if (this.value.toUpperCase() == "POST" && $("#apiForm > select#function").val().replace('/', '') == "run") {
      $("#apiForm > textarea#body").val(JSON.stringify(model_config));
    }
    else if (this.value.toUpperCase() == "GET") {
      $("#apiForm > textarea#body").val("");
    }
  });

  $("textarea#body").on('input', (event) => {
    setTimeout(() => {
      $(event.target).json_beautify();
      autoGrow(event.target);
    }, 5000);
    if ($("#taug-container").length == 0) {
      try {
        $("#taug-container").remove();
      } catch (e) {
        // pass
      }
      $("#apiFormContainer").append('<div id="taug-container"><label for="taug">Tau G:</label><input type="range" id="taug" min="0.01" max="5.0" step="0.01"></div>');
      $("#taug").on("input", function () {
        model_config.model_config.taug = parseFloat($("#taug").val());
        $("#apiForm > textarea#body").val(JSON.stringify(model_config));
      });
    }
  });
  $("textarea#body").json_beautify();
}


// Call the function to generate the API form
(async function () {
  return new Promise(async (resolve, reject) => {
    await generateApiForm();
    // Retrieve the input values and set global variables
    apiKey = document.getElementById('apiKey').value;
    optionalParams = document.getElementById('optionalParams').value;
    body = document.getElementById('body').value;
    selectedFunction = document.getElementById('function').value;
    selectedMethod = document.getElementById('method').value;
    resolve();
  });
})();

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

var chart = null;

const funcMap = {
  'run-at-time': 'runAtTime'
}


const initializeApp = async () => {

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
      console.log(selectedFunction, selectedMethod, optionalParams, body);
      selectedFunction = selectedFunction.replace('/', '');
      if (Object.keys(funcMap).includes(selectedFunction)) {
        selectedFunction = funcMap[selectedFunction];
      }
      selectedMethod = selectedMethod.slice(0, 1).toUpperCase() + selectedMethod.slice(1).toLowerCase();
      var content_type = 'application/json';
      body = body ? body : '{}';
      if (selectedMethod.toUpperCase() == 'GET') {
        body = null;
        content_type = 'application/x-www-form-urlencoded';
      }
      var result = await apigClient[selectedFunction + selectedMethod](optionalParams, body, {
        headers: {
          Authorization: 'Bearer allow',
          Accept: '*/*',
          'Content-Type': content_type
        },
        timeout: 30000
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
  app = await initializeApp();

  // Get the apiFormContainer div element
  var apiFormContainerDiv = document.getElementById("apiFormContainer");

  // Get the restore button element
  var restoreButton = document.createElement("button");
  restoreButton.innerText = "Toggle API Form";
  restoreButton.classList.add("btn", "btn-primary", "mt-2");
  $("ul.navbar-nav").append(`<li id="restore-button-container" class="nav-item"></li>`);
  document.getElementById("restore-button-container").appendChild(restoreButton);

  // Add a click event listener to toggle the class "minimized" on the apiFormContainer div when the div or restore button is clicked
  function toggleMinimized() {
    apiFormContainerDiv.classList.toggle("minimized");
    if (apiFormContainerDiv.classList.contains("minimized")) {
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