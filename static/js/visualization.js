let visualizationChart = null;

function updateChart(data) {
    const ctx = document.getElementById('visualizationChart').getContext('2d');

    if (visualizationChart) {
        visualizationChart.destroy();
    }

    const coordinates = data.coordinates;
    const filenames = data.filenames;

    visualizationChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Audio Files',
                data: coordinates.map((coord, idx) => ({
                    x: coord[0],
                    y: coord[1],
                    filename: filenames[idx]
                })),
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    type: 'linear',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${point.filename} (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
                        }
                    }
                }
            }
        }
    });
}