// In RunAtTimeChart.tsx
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from "react-chartjs-2";

// Register required components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export const RunAtTimeChart: React.FC<{ chartData: any }> = ({ chartData }) => {
  const colors = ["rgb(75,192,192)", "rgb(255,99,132)", "rgb(54,162,235)", "rgb(255,206,86)"];

  const datasets = Object.keys(chartData).map((series, i) => ({
    label: series,
    data: chartData[series].x.map((x: number, idx: number) => ({ x, y: chartData[series].y[idx] })),
    borderColor: colors[i % colors.length],
    fill: false,
    showLine: true
  }));

  return (
    <Line
      data={{ datasets }}
      options={{
        responsive: true,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'Time (hours)' } },
          y: { type: 'linear', title: { display: true, text: 'Value' } }
        }
      }}
    />
  );
};
