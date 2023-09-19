async function setupResizeHandle() {
    const topRow = $('#topRow');
    const bottomRow = $('#bottomRow');
    const containerDiv = $('#containerDiv');
    const resizableHandle = $('#resizableHandle');
    const topRowHeight = topRow.height();
    const bottomRowHeight = bottomRow.height();
    const containerDivHeight = containerDiv.height();
    const resizableHandleHeight = resizableHandle.height();
    const originalTop = resizableHandle.position().top;
    const original_fontsize = $("#resizableHandle").css('font-size');
    const original_margin_top = $("#bottomRow").css('margin-top');
    const original_padding = $("#bottomRow").css('padding');
    const scrollPosition = window.scrollY;
    const topRowChildren = $("#topRow").find("#apiForm,#apiFormContainer");
    resizableHandle.draggable({
        axis: 'y',
        containment: '#containerDiv',
        start: function (event, ui) {
            initialTop = ui.position.top;
        },
        drag: function (event, ui) {
            const mouseY = event.clientY;
            const containerOffset = Math.min(containerDiv.offset().top, window.innerHeight);
            const newTopHeight = Math.min(mouseY - containerOffset * 1.1, window.innerHeight);
            const newBottomHeight = Math.min(containerDiv.height() - newTopHeight, window.innerHeight);
            if ($("#body").position().top + $("#body").height() < mouseY) {
                $("#body").height($("#body").position().top - mouseY);
            }

            if (newTopHeight >= 0 && newBottomHeight >= 0) {
                topRow.height(newTopHeight);
                topRowChildren.height(newTopHeight);
                bottomRow.height(newBottomHeight);

                // Update handle position to match current mouse Y-coordinate
                ui.position.top = 0;
                const currentHeight = resizableHandle.height();
                const movement = Math.min(Math.abs(mouseY - originalTop) / window.innerHeight, 1);
                const newHeight = Math.max(30, Math.min(currentHeight * movement, 70));
                resizableHandle.css('height', `${newHeight}px`);
            } else {
                // Prevent the handle from going out of bounds
                return false;
            }
        }
    });

    $("#resizableHandle").on("mousedown", function (event) {
        event.preventDefault();
        $('*').css('cursor', 'grabbing');
        resizableHandle.css('font-size', '20px');
        $("#bottomRow").css("margin-top", "0px");
        $("#bottomRow").css("padding", "0px !important");
        $("#topRow").css("opacity", "0.5");
        $("#bottomRow").css("position", "sticky");
        $("#bottomRow").css("z-index", "2");
    });
    $("*").on("mouseup", function (event) {
        event.preventDefault();
        $('*').css('cursor', 'default');
        resizableHandle.css('cursor', 'grab');
        resizableHandle.css('font-size', original_fontsize);
        resizableHandle.css('height', `${resizableHandleHeight}px`);
        $("#topRow").css("opacity", "1");
        $("#bottomRow").css("position", "static");
    });
    document.addEventListener("mouseleave", function (event) {
        if (event.clientY <= 0) {
            // indicate too low
            resizableHandle.css('border-top', 'red solid 3px !important');
        }
        else if (event.clientY >= window.innerHeight) {
            // indicate too high
            resizableHandle.css('border-bottom', 'red solid 3px !important');
        }
    });
    document.addEventListener("mouseenter", function (event) {
        resizableHandle.css('border-top', 'none !important');
        resizableHandle.css('border-bottom', 'none !important');
    });
};
