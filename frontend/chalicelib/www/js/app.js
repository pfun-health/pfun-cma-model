// Initialize global variables

var apigClient = null;
var app = {};

// input values
var apiKey = null;
var optionalParams = null;
var body = null;
var selectedFunction = null;
var selectedMethod = null;

async function getRoutesData() {
  // Get the routes data from the /routes endpoint
  var routes_url = window.location.origin + '/routes';
  await axios.get(routes_url, {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
  }).catch((error) => {
    route_url = window.location.origin + '/api/routes';
  });
  return await axios.get(routes_url, {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
  }).then((response) => {
    return response.data;
  }).catch((error) => {
    console.error('Failed to get routes data because: ', error);
  });
}

// Create an HTML form in your UI to input the apiKey, optional parameters, body, function selection, and method selection.
// Function to generate the API form based on the JSON response
async function generateApiForm() {

  // Example JSON response from /routes endpoint
  const routesData = await getRoutesData();

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
  submitButton.classList.add("btn", "btn-primary", "m-2");
  $("ul.navbar-nav").append(`<li id="submit-button-container" class="nav-item"></li>`);
  var secondarySubmitButton = document.createElement("button");
  secondarySubmitButton.setAttribute("id", "secondary-submit-button");
  secondarySubmitButton.setAttribute("type", "button");
  secondarySubmitButton.textContent = "Submit";
  secondarySubmitButton.classList.add("btn", "btn-primary", "m-2");
  secondarySubmitButton.addEventListener("click", function (event) {
    submitButton.click();
  })
  document.getElementById("submit-button-container").appendChild(secondarySubmitButton);

  // get the API key from local storage
  if (localStorage.getItem('PFUN_CMA_API_KEY')) {
    apiKeyInput.value = localStorage.getItem('PFUN_CMA_API_KEY');
  }

  // styling, final stuff...

  // Apply Bootstrap form-control class to input, select, and textarea
  $("#apiForm input, #apiForm select, #apiForm textarea").addClass("form-control");

  // Apply Bootstrap form-group for spacing
  $("#apiForm label, #apiForm input, #apiForm select, #apiForm textarea, #apiForm button").wrapAll("<div class='form-group'></div>");

  // Apply Bootstrap btn and btn-primary classes to the button
  $("#apiForm button").addClass("btn btn-primary");

  // Set default values
  $("select#function").val("/routes");
  $("select#method").val("GET");

  // Event listeners
  $("select#function").on("change", function () {
    if (this.value == "run" && $("select#method").val().toUpperCase() == "POST") {
      $("textarea#body").val(JSON.stringify(model_config));
    }
  });

  $("select#method").on("change", function () {
    if (this.value.toUpperCase() == "POST" && $("select#function").val().replace('/', '') == "run") {
      $("textarea#body").val(JSON.stringify(model_config));
    }
    else if (this.value.toUpperCase() == "GET") {
      $("textarea#body").val("");
    }
  });

  $("textarea#body").on('blur', (event) => {
    setTimeout(() => {
      $(event.target).json_beautify();
      autoGrow(event.target);
    }, 500);
    if ($("#taug-container").length == 0) {
      try {
        $("#taug-container").remove();
      } catch (e) {
        // pass
      }
      $("#apiFormContainer").append('<div id="taug-container"><label for="taug">Tau G:</label><input type="range" id="taug" min="0.01" max="5.0" step="0.01"></div>');
      $("#taug").on("input", function () {
        model_config.model_config.taug = parseFloat($("#taug").val());
        $("textarea#body").val(JSON.stringify(model_config));
      });
    }
  });
  $("textarea#body").json_beautify();
}


// Call the function to generate the API form
(async function () {
  return new Promise(async (resolve, reject) => {
    await generateApiForm();
    // Set default values
    $("select#function").val("/routes");
    $("select#method").val("GET");
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
    t: null,
    N: 24,
    d: 0,
    taup: 1,
    taug: 1,
    B: 0.05,
    Cm: 0,
    toff: 0,
    tM: [7, 11, 17.5],
    seed: null,
    eps: 1e-18,
  }
};

var chart = null;

function simulateDrag(dy = null) {
  const resizableHandle = $("#resizableHandle");
  const outputBottom = $("#output-area").position().top;
  if (dy === null) {
    // update the UI to make room for the output
    dy = $("#output-area").height() + outputBottom - $("nav").position().top;
  }
  resizableHandle.simulate("drag", {
    dx: resizableHandle.offset().left,
    dy: -dy,
  });
};


const funcMap = {
  'run-at-time': 'runAtTime',
  'run': 'run',
  'fit': 'fit'
}

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

  // prepare the payload
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

    // make room for the output
    simulateDrag();

    // make the request
    var result = await apigClient[selectedFunction + selectedMethod](optionalParams, body, {
      headers: {
        Authorization: 'Bearer allow',
        Accept: '*/*',
        'Content-Type': content_type
      },
      timeout: 60000
    }).catch(function (err) {
      console.warn(`failed to access the specified endpoint: '${selectedFunction}${selectedMethod}'.\nError:`, err);

    })

    // set raw json data...
    var data = result.data;
    var string_data = JSON.stringify(data);
    console.log(string_data);
    $("#output-area").html(
      `<code>${string_data}</code>`
    );

    // update chart...
    const ctx = document.getElementById("chart");
    if (chart) {
      chart.clear();
      chart.destroy();
    }
    ctx.style.height = '0px';
    if (selectedFunction == 'run') {
      var arr = [];
      var Carr = [];
      var Marr = [];
      var Aarr = [];
      const G = Object.values(data.G);
      const C = Object.values(data.c);
      const M = Object.values(data.m);
      const A = Object.values(data.a);
      Object.values(data.t).forEach((value, index) => {
        arr.push({
          x: value,
          y: G[index]
        });
        Carr.push({
          x: value,
          y: C[index]
        });
        Marr.push({
          x: value,
          y: M[index]
        });
        Aarr.push({
          x: value,
          y: A[index]
        })
      });
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
            },
            {
              label: 'Melatonin',
              data: Marr
            },
            {
              label: 'Adiponectin',
              data: Aarr
            }
          ]
        }
      });
    }

  } catch (err) {
    console.warn(`failed to access the specified endpoint: '${selectedFunction}${selectedMethod}'.\nError:`, err);
    throw err;
  }
  // make room for the output (after request)
  simulateDrag();
}

const initializeApp = async () => {
// Initialize the app.

  // Add event listeners to handle form submission.
  document.getElementById('apiForm').addEventListener('submit', async (event) => {
    await progressIndicator(submitFunction, event);
  });

  return app;
};

async function setupApp() {
  app = await initializeApp();
  $("select#function").val("/routes");
  $("select#method").val("GET");

  // setup the route links
  async function setupRouteLinks() {

    // Select all "a" elements with the class "route-link"
    const links = document.querySelectorAll('a.route-link');

    // Add event listener to each link
    links.forEach(link => {
      link.addEventListener('click', async function (event) {
        event.preventDefault(); // Prevent the default action
        const routeName = this.id; // Get the ID of the clicked element

        // Call the corresponding function from apigClient
        if (routeName) {
          const routeData = await getRoutesData();
          var routeMethod = routeData[routeName][0];
          if (["/run", "/run-at-time"].includes(routeName)) {
            routeMethod = 'POST';
          }
          $("select#function").val(routeName);
          $("select#method").val(routeMethod);
          $("button[type=submit]").click();
          // console.log(routeData);
          // console.log(routeName, routeMethod);
          // console.log(routeData[routeName][0]);
        } else {
          console.warn(`Route name not found or not a function: ${routeName}`);
        }
      });
    });
  }

  // Run the function to set up the event listeners
  await setupRouteLinks();

}

setupApp();