
const showLoadingContainer = async () => {
    const loadingContainer = $("#loading-container");
    $(loadingContainer).removeClass("d-none");
    $(loadingContainer).show();
};

const showAlerts = async () => {
    const error_msg = "Whoops! The server is busy right now."
    "\nRetrying your request... Please wait.";
    const timeTillHide = 4500; // time in ms until the toast should hide
    const errToast = $.toast({
	heading: 'Error',
	text: `${error_msg}`,
	showHideTransition: 'fade',
	icon: 'error',
	hideAfter: timeTillHide,
	stack: true,
    });
    setTimeout(() => {
	errToast.update({
	    heading: 'Trying again...',
	    text: 'Attempting your request again... Please wait.',
	    hideAfter: timeTillHide,
	});
	setTimeout(async () => {
	    // try the request again.
	    await onFormSubmit();
	}, timeTillHide + 500); // add a little buffer time after the toast hides to try the request again
    }, timeTillHide + 1230);``
};
    
const onFormSubmit = async (event) => {

    try {
        event.preventDefault();
    } catch(err) {
        console.warn(err);
    }

    // show loading indicator
    showLoadingContainer();

    // get the relevant elements
    const query = document.getElementById('query-input').value;
    const query_url = new URL(window.location.origin + "/llm/generate-scenario");
    const responseOutput = $("#response-output");
    const fmtOutput = $("#formatted-response-output");

    try {
        if(query === '<<TEST_ERROR>>') {
            // test the error behavior (not a real prompt)
            throw new Error("<<TEST_ERROR>>");
        }
        query_url.searchParams.set('prompt', query);
        $("#submit-btn").addClass("disabled"); // disable the submit button temporarily
	$(fmtOutput).html(''); // clear the existing recommendations
	$(responseOutput).html(''); // clear old response data
        const response = await fetch(query_url.toString(), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: null
        }).then((response) => {
            console.log('raw response:\n', response);
            $("#loading-container").hide();
            return response;
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }

        const data = await response.json();
        const str_content = JSON.stringify(data, null, 2);
        if(!str_content.startsWith("Err")) {
            console.debug('confirmed! this is a successful response');
            // update the response output (in the UI)
            let recs_data = data.recommendations;
            console.log("Recommendations data:", recs_data);
            $(Object.entries(recs_data)).each((ix, [key, value]) => {
                $(fmtOutput).append(
                    `<dt>${key}</dt><dd>${value}</dd>`
                );  // update the formatted response output
            });
            $(responseOutput).text(str_content); // update the raw response output text
	    // scroll the output area into view
	    // Source - https://stackoverflow.com/a/20760191
	    // Posted by Atharva, modified by community. See post 'Timeline' for change history
	    // Retrieved 2026-02-16, License - CC BY-SA 4.0
	    $("#output-title").get(0).scrollIntoView({behavior: 'smooth'});

            // re-enable the submit button
            $("#submit-btn").removeClass("disabled");
        } else {
            // there was an error otherwise
            throw new Error(str_content);
        }
    } catch (error) {
        // ensure the user can't submit manually again during this time
        $("#submit-btn").addClass("disabled");
        // increment the count of retries
        let ntry_count = parseInt(localStorage.getItem("ntry_count"));
        ntry_count = ntry_count + 1;
        localStorage.setItem("ntry_count", `${ntry_count}`);
        if(ntry_count >= 4) {
            alert(
                "Your request has been tried too many times.\n\nAfter the page reloads, please try again." +
                "\nThe page should automatically refresh after you close this dialog."
            );
            // attempt to refresh the page automatically
            window.location.reload();
        }
        console.warn(`Error: ${error.message}`);
	// show the alerts as toasts
	await showAlerts();
    }
};

const setupJqToast = async () => {
    const jqToast = await import("/static/js/jquery.toast.min.js").then(
	(Module) => { return Module.default; });
    jqToast(jQuery, window, document);
    console.debug('...finished setup for jquery toast, try: $.toast(...)');
};

document.addEventListener("DOMContentLoaded", () => {
    localStorage.setItem('ntry_count', '0'); // number of times we've tried the endpoint before reloading
    // setup form submission handler
    document.getElementById('query-form').addEventListener('submit', onFormSubmit);
    // import, configure the jqToast module
    setupJqToast();
});
