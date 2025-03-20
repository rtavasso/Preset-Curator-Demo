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

/**
 * Vital Curator Application
 * 
 * This module handles the client-side functionality for the audio similarity application.
 * It manages audio playback, feature comparison, and dynamic UI updates.
 */
class AudioSimilarityApp {
    /**
     * Initialize the application
     */
    constructor() {
        // DOM Elements
        this.elements = {
            queryForm: document.getElementById('queryForm'),
            audioSelect: document.getElementById('audioSelect'),
            featureWeights: document.querySelectorAll('.feature-weight'),
            resultsContainer: document.getElementById('results'),
            resultsHeader: document.querySelector('.card-header .card-title')
        };
        
        // Get references to the audio player elements
        this.elements.audioPlayer = this.elements.queryForm.querySelector('audio');
        this.elements.playButton = this.elements.queryForm.querySelector('.custom-play-button');
        
        // App state
        this.state = {
            currentAudioFile: null,
            currentlyPlaying: null,
            databaseStats: null,
            allFeatures: {},
            queryFeatures: null,
            neighborFeatures: {},
            currentNeighbors: []
        };
        
        // Throttled/debounced functions
        this.debouncedRerank = this.debounce(this.rerankLocally.bind(this), 250);
        
        // Initialize the application
        this.init();
    }
    
    /**
     * Debounce function to limit how often a function is called
     */
    debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
    
    /**
     * Initialize the application
     */
    init() {
        // Check database status on load
        this.checkDatabaseStatus();
        
        // Initialize all existing audio players
        this.initializeAudioPlayers();
        
        // Add event listeners
        this.setupEventListeners();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Audio file selection
        this.elements.audioSelect.addEventListener('change', this.handleAudioSelection.bind(this));
        
        // Feature weight sliders
        this.elements.featureWeights.forEach(slider => {
            slider.addEventListener('input', this.debouncedRerank);
        });
        
        // Form submission (kept for backward compatibility)
        this.elements.queryForm.addEventListener('submit', (e) => {
            e.preventDefault();
            // Form submission is handled automatically by selection change
        });
    }
    
    /**
     * Handle audio selection change
     */
    handleAudioSelection(event) {
        const value = event.target.value;
        
        if (value) {
            // Update audio source
            this.elements.audioPlayer.src = `/audio/${value}`;
            
            // Enable play button
            this.elements.playButton.disabled = false;
            
            // Reset the player UI
            const playerContainer = this.elements.playButton.closest('.custom-audio-player');
            this.resetPlayerUI(playerContainer);
            
            // Auto-compare when a new file is selected
            this.state.currentAudioFile = value;
            this.getInitialComparison();
        } else {
            // Disable play button when no file is selected
            this.elements.playButton.disabled = true;
        }
    }
    
    /**
     * Check database status
     */
    async checkDatabaseStatus() {
        try {
            const response = await fetch('/database-stats');
            if (!response.ok) {
                console.error('Failed to fetch database stats');
                return;
            }
            
            this.state.databaseStats = await response.json();
            this.updateDatabaseStatusUI();
        } catch (error) {
            console.error('Error checking database status:', error);
        }
    }
    
    /**
     * Update the database status UI
     */
    updateDatabaseStatusUI() {
        const { databaseStats } = this.state;
        if (!databaseStats) return;
        
        const dbBadge = document.querySelector('.badge.bg-primary');
        if (!dbBadge) return;
        
        if (databaseStats.initialized) {
            dbBadge.textContent = `Fast Database (${databaseStats.num_files} files)`;
            dbBadge.classList.remove('bg-secondary');
            dbBadge.classList.add('bg-success');
            
            // Add tooltip with database stats
            dbBadge.setAttribute('data-bs-toggle', 'tooltip');
            dbBadge.setAttribute('data-bs-placement', 'left');
            dbBadge.setAttribute('title', 
                `Initialized in ${databaseStats.initialization_seconds}s with ${databaseStats.num_features} features`);
            
            // Initialize tooltip
            new bootstrap.Tooltip(dbBadge);
        } else {
            dbBadge.textContent = 'Database Disabled';
            dbBadge.classList.remove('bg-primary');
            dbBadge.classList.add('bg-secondary');
        }
        
        console.log('Database stats:', databaseStats);
    }
    
    /**
     * Get initial comparison data
     */
    async getInitialComparison() {
        // Show loading indicator
        this.showLoadingState();
        
        // Reset data
        this.state.allFeatures = {};
        this.state.queryFeatures = null;
        this.state.neighborFeatures = {};
        this.state.currentNeighbors = [];
        
        // Collect current weights
        const weights = this.getCurrentWeights();

        try {
            const startTime = performance.now();
            
            const response = await fetch('/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    queryFile: this.state.currentAudioFile,
                    weights: weights
                })
            });

            const responseData = await response.json();
            
            const endTime = performance.now();
            const queryTime = ((endTime - startTime) / 1000).toFixed(2);
            
            if (responseData.error) {
                this.showError(responseData.error);
                return;
            }

            // Store feature data for client-side reranking
            this.processComparisonResponse(responseData, queryTime);
            
        } catch (error) {
            console.error('Comparison error:', error);
            this.showError('Error comparing files');
        }
    }
    
    /**
     * Process the comparison response data
     */
    processComparisonResponse(responseData, queryTime) {
        if (responseData.all_features) {
            // Store the features for reranking
            this.state.allFeatures = responseData.all_features;
            this.state.queryFeatures = responseData.all_features.query;
            
            // Process similarity data
            const similarities = responseData.similarities || [];
            this.state.currentNeighbors = similarities.map(sim => 
                sim.filename.replace('.wav', ''));
            
            // Create mapping of neighbor features
            this.state.currentNeighbors.forEach(neighbor => {
                if (responseData.all_features[neighbor]) {
                    this.state.neighborFeatures[neighbor] = responseData.all_features[neighbor];
                }
            });
        }
        
        // Display initial results
        const similarities = responseData.similarities || [];
        
        // Add query time to results header
        if (this.elements.resultsHeader) {
            this.elements.resultsHeader.innerHTML = 
                `Similar Patches <small class="text-muted" style="font-size: 0.8rem; margin-left: 8px;">(retrieved in ${queryTime}s)</small>`;
        }
        
        this.displayResults(similarities);
    }
    
    /**
     * Show loading state
     */
    showLoadingState() {
        const { resultsContainer, databaseStats } = this.state;
        
        // Use different loading message based on database status
        if (this.state.databaseStats && this.state.databaseStats.initialized) {
            this.elements.resultsContainer.innerHTML = 
                '<div class="col-12 text-center"><div class="spinner-border text-success" role="status"></div><p class="mt-2">Retrieving from database...</p></div>';
        } else {
            this.elements.resultsContainer.innerHTML = 
                '<div class="col-12 text-center"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Calculating similarities...</p></div>';
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        alert(message);
        this.elements.resultsContainer.innerHTML = 
            `<div class="col-12 text-center text-danger">Error: ${message}</div>`;
    }
    
    /**
     * Get current feature weights
     */
    getCurrentWeights() {
        const weights = {};
        this.elements.featureWeights.forEach(slider => {
            weights[slider.id] = parseInt(slider.value);
        });
        return weights;
    }
    
    /**
     * Perform client-side reranking
     */
    rerankLocally() {
        const { queryFeatures, neighborFeatures, currentNeighbors } = this.state;
        
        if (!queryFeatures || Object.keys(neighborFeatures).length === 0) {
            console.error('Missing feature data for reranking');
            return;
        }
        
        // Show reranking indicator
        this.elements.resultsContainer.classList.add('reranking');
        
        // Get current weights
        const weights = this.getCurrentWeights();
        
        // Calculate reranked results
        const results = this.calculateLocalRanking(weights);
        
        // Display results
        this.displayResults(results);
        
        // Update the header to show this is a client-side ranking
        if (this.elements.resultsHeader) {
            this.elements.resultsHeader.innerHTML = 
                `Similar Patches <small class="text-muted" style="font-size: 0.8rem; margin-left: 8px;">(reranked locally)</small>`;
        }
        
        // Remove reranking indicator
        this.elements.resultsContainer.classList.remove('reranking');
    }
    
    /**
     * Calculate local ranking based on current weights
     */
    calculateLocalRanking(weights) {
        const { queryFeatures, neighborFeatures, currentNeighbors } = this.state;
        
        // Filter active weights
        const activeWeights = {};
        let hasActiveWeights = false;
        for (const [feature, weight] of Object.entries(weights)) {
            if (weight !== 0) {
                activeWeights[feature] = weight;
                hasActiveWeights = true;
            }
        }
        
        // Calculate standard scaler values
        const { featureStats, scaledValues } = this.calculateStandardScaler(activeWeights);
        
        // Calculate similarities and feature deltas
        const results = this.calculateFeatureDeltas(
            activeWeights, hasActiveWeights, scaledValues
        );
        
        // Sort by weighted score (higher is better)
        results.sort((a, b) => b.weighted_score - a.weighted_score);
        
        // Remove the weighted_score field from results
        for (const result of results) {
            delete result.weighted_score;
        }
        
        return results;
    }
    
    /**
     * Calculate standard scaler for features
     */
    calculateStandardScaler(activeWeights) {
        const { queryFeatures, neighborFeatures, currentNeighbors } = this.state;
        
        // Perform client-side StandardScaler for all features in weights
        const featureValues = {};
        const featureSources = [...currentNeighbors, 'query'];
        
        // First, collect all values for each feature
        for (const feature in activeWeights) {
            featureValues[feature] = [];
            
            // Add all neighbor values first
            for (const neighbor of currentNeighbors) {
                if (neighborFeatures[neighbor] && 
                    feature in neighborFeatures[neighbor] && 
                    !Array.isArray(neighborFeatures[neighbor][feature])) {
                    featureValues[feature].push(neighborFeatures[neighbor][feature]);
                }
            }
            
            // Add query value
            if (queryFeatures && feature in queryFeatures && !Array.isArray(queryFeatures[feature])) {
                featureValues[feature].push(queryFeatures[feature]);
            }
        }
        
        // Calculate mean and std for each feature (StandardScaler)
        const featureStats = {};
        for (const [feature, values] of Object.entries(featureValues)) {
            if (values.length > 1) {
                const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
                const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
                const std = Math.sqrt(variance) || 1;  // Avoid divide by zero
                
                featureStats[feature] = { mean, std };
            }
        }
        
        // Calculate scaled values
        const scaledValues = {
            query: {}
        };
        
        // Scale query values
        for (const [feature, stats] of Object.entries(featureStats)) {
            if (queryFeatures && feature in queryFeatures) {
                scaledValues.query[feature] = (queryFeatures[feature] - stats.mean) / stats.std;
            }
        }
        
        // Scale neighbor values
        for (const neighbor of currentNeighbors) {
            scaledValues[neighbor] = {};
            
            for (const [feature, stats] of Object.entries(featureStats)) {
                if (neighborFeatures[neighbor] && feature in neighborFeatures[neighbor]) {
                    scaledValues[neighbor][feature] = 
                        (neighborFeatures[neighbor][feature] - stats.mean) / stats.std;
                }
            }
        }
        
        return { featureStats, scaledValues };
    }
    
    /**
     * Calculate feature deltas and similarity scores
     */
    calculateFeatureDeltas(activeWeights, hasActiveWeights, scaledValues) {
        const { currentNeighbors } = this.state;
        const results = [];
        
        for (const neighbor of currentNeighbors) {
            // Calculate feature deltas and weighted score
            const featureDeltas = {};
            let weightedScore = 0;
            let totalRawDistance = 0;
            let featureCount = 0;
            
            const totalWeight = hasActiveWeights ? 
                Object.values(activeWeights).reduce((sum, w) => sum + Math.abs(w), 0) : 1;
            
            for (const [feature, weight] of Object.entries(activeWeights)) {
                if (weight === 0) continue;
                
                if (scaledValues.query[feature] !== undefined && 
                    scaledValues[neighbor][feature] !== undefined) {
                    
                    // Delta in standard deviations
                    const delta = scaledValues[neighbor][feature] - scaledValues.query[feature];
                    
                    // Convert to percentage (1 std = 100%)
                    const deltaPercent = delta * 100;
                    featureDeltas[feature] = {
                        delta: deltaPercent,
                        weight: weight
                    };
                    
                    // Calculate weighted score for similarity
                    const normWeight = Math.abs(weight) / totalWeight;
                    
                    if (weight > 0) {
                        // Higher values are better
                        weightedScore += normWeight * delta;
                    } else {
                        // Lower values are better
                        weightedScore -= normWeight * delta;
                    }
                    
                    // Add to raw distance calculation (for distance metric)
                    totalRawDistance += delta * delta; // Squared difference
                    featureCount++;
                }
            }
            
            // SIMILARITY: Calculate based on how well the neighbor matches preferences (0-100%)
            let similarity = 0;
            if (hasActiveWeights) {
                // Use sigmoid function to map weighted scores to a 0-1 range
                // Using a sigmoid centered at 0 with a steepness factor
                similarity = 1 / (1 + Math.exp(-weightedScore * 1.5));
            } else {
                // Default similarity when no weights are active
                similarity = 0.5;  // 50%
            }
            
            // DISTANCE: Calculate the Euclidean distance in z-score space
            let distance = 0;
            if (featureCount > 0) {
                distance = Math.sqrt(totalRawDistance / featureCount);
            }
            
            results.push({
                filename: neighbor + '.wav',  // Add .wav extension for consistency
                similarity: similarity,
                distance: distance,
                weighted_score: weightedScore,
                feature_deltas: featureDeltas
            });
        }
        
        return results;
    }
    
    /**
     * Display results in the UI
     */
    displayResults(similarities) {
        const { resultsContainer } = this.elements;
        resultsContainer.innerHTML = '';

        const template = document.getElementById('resultCardTemplate');

        similarities.forEach((result, index) => {
            const clone = template.content.cloneNode(true);
            const card = clone.querySelector('.card');

            // Set basic info
            card.querySelector('.card-title').textContent = result.filename.replace('.wav', '');
            
            // Ensure similarity is always a value between 0-100%
            const similarity = result.similarity;
            let displaySimilarity;
            
            if (typeof similarity === 'number' && similarity >= 0 && similarity <= 1) {
                // If it's already in the 0-1 range, convert to percentage
                displaySimilarity = (similarity * 100).toFixed(2) + '%';
            } else if (typeof similarity === 'number') {
                // If it's outside the 0-1 range, clamp it
                const clampedValue = Math.max(0, Math.min(1, (similarity + 1) / 2));
                displaySimilarity = (clampedValue * 100).toFixed(2) + '%';
            } else {
                // Fallback if similarity is not a valid number
                displaySimilarity = "N/A";
            }
            
            card.querySelector('.similarity').textContent = displaySimilarity;
            
            // Format distance - now shows actual standard deviation distance
            const displayDistance = typeof result.distance === 'number' 
                ? result.distance.toFixed(2) + ' σ' // Display as standard deviations
                : "N/A";
            card.querySelector('.distance').textContent = displayDistance;

            // Set audio source
            const audio = card.querySelector('audio');
            const source = audio.querySelector('source');
            source.src = `/audio/${result.filename}`;
            audio.load(); // Force the audio to load the new source
            
            // Setup custom audio player
            const playerContainer = card.querySelector('.custom-audio-player');
            this.setupAudioPlayer(playerContainer);
            
            // Setup feature comparison bars
            this.setupFeatureBars(card, result);

            resultsContainer.appendChild(clone);
        });
        
        // Initialize icons
        feather.replace();
        
        // Initialize tooltips
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
    }
    
    /**
     * Setup feature comparison bars
     */
    setupFeatureBars(card, result) {
        const featureBars = card.querySelector('.feature-bars');
        
        // Clear any existing feature bars
        featureBars.innerHTML = '';
        
        // Get weights
        const weights = this.getCurrentWeights();
        
        // Filter and sort feature deltas to only show non-zero weights
        const nonZeroFeatures = Object.entries(result.feature_deltas || {})
            .filter(([_, delta]) => delta.weight !== 0)
            .sort((a, b) => a[0].localeCompare(b[0]));
        
        if (nonZeroFeatures.length === 0) {
            // If no non-zero weights, show a message
            featureBars.innerHTML = '<div class="text-muted small fst-italic">Adjust feature weights to see comparison</div>';
            return;
        }
        
        // Create feature bars for non-zero weighted features
        for (const [feature, delta] of nonZeroFeatures) {
            this.createFeatureBar(featureBars, feature, delta);
        }
    }
    
    /**
     * Create a single feature comparison bar
     */
    createFeatureBar(featureBars, feature, delta) {
        const barContainer = document.createElement('div');
        barContainer.className = 'feature-bar-container mb-2';

        const label = document.createElement('div');
        label.className = 'feature-label d-flex justify-content-between';
        const featureName = feature.charAt(0).toUpperCase() + feature.slice(1); // Capitalize first letter
        
        // Get the weight value to display
        const weight = delta.weight;
        const weightIndicator = `<small class="weight-indicator ${weight > 0 ? 'weight-positive' : 'weight-negative'}">${weight > 0 ? '+' : ''}${weight}</small>`;
        
        // Ensure delta is within a reasonable range (-100 to 100)
        const displayDelta = Math.max(Math.min(delta.delta, 100), -100);
        
        label.innerHTML = `
            <span>${featureName} ${weightIndicator}</span>
            <span class="delta-value">${displayDelta > 0 ? '+' : ''}${displayDelta.toFixed(2)}%</span>
        `;

        const barWrapper = document.createElement('div');
        barWrapper.className = 'bar-wrapper position-relative';
        
        // Add weight emphasis indicator
        barWrapper.classList.add('weighted');
        barWrapper.classList.add(weight > 0 ? 'weight-positive' : 'weight-negative');
        barWrapper.style.opacity = Math.min(1.0, 0.7 + (Math.abs(weight) / 10));

        const bar = document.createElement('div');
        bar.className = `progress-bar ${delta.delta < 0 ? 'bg-danger' : 'bg-success'}`;

        // Calculate bar width - use displayDelta to stay within reasonable limits
        const absPercentage = Math.abs(displayDelta);
        
        // Position the bar properly for center alignment
        if (delta.delta < 0) {
            // For negative values, place the bar to the left of center
            bar.style.width = `${absPercentage}%`;
            bar.style.position = 'absolute';
            bar.style.right = '50%';
            bar.style.left = 'auto';
        } else {
            // For positive values, place the bar to the right of center
            bar.style.width = `${absPercentage}%`;
            bar.style.position = 'absolute';
            bar.style.left = '50%';
            bar.style.right = 'auto';
        }

        barWrapper.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        featureBars.appendChild(barContainer);
    }
    
    /**
     * Initialize all audio players
     */
    initializeAudioPlayers() {
        document.querySelectorAll('.custom-audio-player').forEach(playerContainer => {
            this.setupAudioPlayer(playerContainer);
        });
    }
    
    /**
     * Setup a single audio player
     */
    setupAudioPlayer(playerContainer) {
        const audio = playerContainer.querySelector('audio');
        const playButton = playerContainer.querySelector('.custom-play-button');
        
        // Initialize play button icon
        this.updatePlayButtonIcon(playButton, false);
        
        // Play button click handler
        playButton.addEventListener('click', () => {
            if (audio.paused) {
                // If another player is playing, stop it first
                if (this.state.currentlyPlaying && this.state.currentlyPlaying !== audio) {
                    this.state.currentlyPlaying.pause();
                    this.state.currentlyPlaying.currentTime = 0;
                    const otherContainer = this.state.currentlyPlaying.closest('.custom-audio-player');
                    if (otherContainer) {
                        otherContainer.classList.remove('playing');
                        this.updatePlayButtonIcon(otherContainer.querySelector('.custom-play-button'), false);
                    }
                }
                
                // Play this audio
                audio.play();
                playerContainer.classList.add('playing');
                this.updatePlayButtonIcon(playButton, true);
                this.state.currentlyPlaying = audio;
            } else {
                // Stop/reset the audio
                audio.pause();
                audio.currentTime = 0;
                playerContainer.classList.remove('playing');
                this.updatePlayButtonIcon(playButton, false);
                this.state.currentlyPlaying = null;
            }
        });
        
        // Reset UI when audio ends
        audio.addEventListener('ended', () => {
            audio.currentTime = 0;
            playerContainer.classList.remove('playing');
            this.updatePlayButtonIcon(playButton, false);
            this.state.currentlyPlaying = null;
        });
    }
    
    /**
     * Update play button icon
     */
    updatePlayButtonIcon(button, isPlaying) {
        // Clear current icon
        button.innerHTML = '';
        
        // Create new icon
        const icon = document.createElement('i');
        icon.setAttribute('data-feather', isPlaying ? 'square' : 'play');
        button.appendChild(icon);
        
        // Initialize the icon
        feather.replace();
    }
    
    /**
     * Reset player UI
     */
    resetPlayerUI(playerContainer) {
        const audio = playerContainer.querySelector('audio');
        const playButton = playerContainer.querySelector('.custom-play-button');
        
        audio.pause();
        audio.currentTime = 0;
        playerContainer.classList.remove('playing');
        this.updatePlayButtonIcon(playButton, false);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AudioSimilarityApp();
});