JSON.parseall = function (text) {
    // Replace NaN with null for JSON parsing
    return JSON.parse(text.replaceAll(/\bNaN\b/g, "null"));
};

class DataRow {
    constructor(data) {
        this.data = data;
    }

    get ts_local() {
        return this.data.ts_local;
    }

    get sg() {
        return this.data.sg;
    }

    get meal_tag() {
        return this.data.meal_tag;
    }

    insertRow(tableBody) {
        const row = tableBody.insertRow();
        const cellTime = row.insertCell();
        const cellGlucose = row.insertCell();
        const cellMealTag = row.insertCell();

        cellTime.textContent = this.ts_local;
        cellGlucose.textContent = this.sg.toFixed(2);
        cellMealTag.textContent = this.meal_tag;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startStream');
    const stopButton = document.getElementById('stopStream');
    const dataBody = document.getElementById('data-body');
    let controller;

    const startStream = async () => {
        controller = new AbortController();
        const signal = controller.signal;

        const pct0 = document.getElementById('pct0').value;
        const nrows = document.getElementById('nrows').value;

        startButton.disabled = true;
        stopButton.disabled = false;
        dataBody.innerHTML = ''; // Clear previous data

        try {
            // Fetch the data stream
            const response = await fetch(
                `/data/sample/stream?pct0=${pct0}&nrows=${nrows}&media_type=octet-stream`,
                { headers: { 'Content-Type': 'application/octet-stream' }, signal }
            );
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            // Function to read and process the stream (async -> formatted table rows)
            const read = async () => {
                const { done, value } = await reader.read();
                if (done) {
                    console.log('Stream complete');
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    return;
                }

                const chunk = decoder.decode(value, { stream: true });
                const rows = chunk.split('\n').filter(row => row.trim() !== '');

                rows.forEach(row => {
                    try {
                        // Replace NaN with null for JSON parsing
                        const data = JSON.parseall(row);
                        const dataRow = new DataRow(data);
                        dataRow.insertRow(dataBody);
                    } catch (e) {
                        console.error('Failed to parse row:', row, e);
                    }
                });    
            };

            // begin reading the stream (async)
            read();
            console.log('Streaming the sample data has started');

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Stream aborted');
            } else {
                console.error('Error fetching stream:', error);
            }
            startButton.disabled = false;
            stopButton.disabled = true;
        }
    };

    const stopStream = () => {
        if (controller) {
            controller.abort();
            console.log('Stopping stream...');
        }
    };

    startButton.addEventListener('click', startStream);
    stopButton.addEventListener('click', stopStream);
});
