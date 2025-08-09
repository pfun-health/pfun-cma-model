// src/utils.ts
export function parseSampleData(data: any) {
  const chartData: { [series: string]: { x: number[]; y: number[] } } = {};

  for (const seriesName of Object.keys(data)) {
    const pointEntries = Object.entries(data[seriesName])
      .map(([xStr, y]) => [parseFloat(xStr), Number(y)] as [number, number])
      .sort((a, b) => a[0] - b[0]); // ensure ascending x order

    const xVals = pointEntries.map(entry => entry[0]);
    const yVals = pointEntries.map(entry => entry[1]);

    chartData[seriesName] = { x: xVals, y: yVals };
  }

  return chartData;
}
