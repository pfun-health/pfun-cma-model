async function setupResizeHandle() {
    const topRow = $('.topRow').first();
    const bottomRow = $('.bottomRow').first();
    const parentRow = topRow.parent();

    $('#resizableHandle').draggable({
        axis: 'y',
        containment: 'parent',
        drag: function (event, ui) {
            const totalHeight = containerDiv.height();
            const topRowHeight = ui.position.top;
            const bottomRowHeight = totalHeight - topRowHeight;

            topRow.css({ height: `${topRowHeight}px` });
            bottomRow.css({ height: `${bottomRowHeight}px` });
        },
    });
};
