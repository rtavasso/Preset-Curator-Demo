/**
 * PROPRIETARY AND CONFIDENTIAL
 *
 * Copyright (c) 2025-2026. All Rights Reserved.
 *
 * NOTICE: All information contained herein is, and remains the property of the owner.
 * The intellectual and technical concepts contained herein are proprietary and may be
 * covered by U.S. and Foreign Patents, patents in process, and are protected by trade
 * secret or copyright law. Dissemination of this information or reproduction of this
 * material is strictly forbidden unless prior written permission is obtained from the
 * owner. Access to the source code contained herein is hereby forbidden to anyone except
 * current employees or contractors of the owner who have executed Confidentiality and
 * Non-disclosure Agreements explicitly covering such access.
 *
 * THE RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT
 * CONVEY OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO
 * MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
 *
 * Usage of this source code is subject to a strict license agreement. Unauthorized
 * reproduction, modification, or distribution, in part or in whole, is strictly prohibited.
 * License terms available upon request.
 */

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