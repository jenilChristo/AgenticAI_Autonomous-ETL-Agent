/* Main JavaScript for DevOps Interface */

// Global configuration
const CONFIG = {
    refreshInterval: 30000, // 30 seconds
    etlStatusCheckInterval: 10000, // 10 seconds
    maxRetries: 3,
    apiEndpoints: {
        etlStatus: '/api/etl-status',
        runEtl: '/api/run-etl',
        projects: '/api/projects',
        stories: '/api/stories',
        tasks: '/api/tasks'
    }
};

// Global state management
const AppState = {
    currentProject: null,
    currentStory: null,
    etlStatus: {},
    refreshTimer: null,
    retryCount: 0
};

// Utility functions
const Utils = {
    // Show alert message
    showAlert: function(message, type = 'info', duration = 5000) {
        const alertId = 'alert-' + Date.now();
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)}"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        const container = document.querySelector('.alert-container') || 
                         document.querySelector('.container-fluid') || 
                         document.body;
        
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = alertHTML;
        container.insertBefore(tempDiv.firstElementChild, container.firstChild);
        
        // Auto-dismiss after duration
        if (duration > 0) {
            setTimeout(() => {
                const alertElement = document.getElementById(alertId);
                if (alertElement) {
                    alertElement.remove();
                }
            }, duration);
        }
        
        return alertId;
    },

    getAlertIcon: function(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle',
            'primary': 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    // Format date/time
    formatDateTime: function(dateString, format = 'short') {
        const date = new Date(dateString);
        const options = {
            short: { 
                month: 'short', 
                day: 'numeric', 
                hour: '2-digit', 
                minute: '2-digit' 
            },
            long: { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric', 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit' 
            },
            date: { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            }
        };
        
        return date.toLocaleDateString('en-US', options[format] || options.short);
    },

    // Format duration
    formatDuration: function(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Copy to clipboard
    copyToClipboard: function(text, successMessage = 'Copied to clipboard') {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(() => {
                this.showAlert(successMessage, 'success', 2000);
            }).catch(() => {
                this.fallbackCopyToClipboard(text, successMessage);
            });
        } else {
            this.fallbackCopyToClipboard(text, successMessage);
        }
    },

    fallbackCopyToClipboard: function(text, successMessage) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.showAlert(successMessage, 'success', 2000);
        } catch (err) {
            this.showAlert('Failed to copy to clipboard', 'danger', 3000);
        }
        
        document.body.removeChild(textArea);
    }
};

// API handler
const API = {
    request: async function(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'same-origin'
        };

        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    },

    get: function(url) {
        return this.request(url, { method: 'GET' });
    },

    post: function(url, data) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    put: function(url, data) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    delete: function(url) {
        return this.request(url, { method: 'DELETE' });
    }
};

// ETL Agent management
const ETLAgent = {
    runAgent: async function(storyId, showProgress = true) {
        const button = event?.target?.closest('button');
        let originalContent = '';
        
        if (button && showProgress) {
            originalContent = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            button.disabled = true;
        }

        try {
            const response = await API.post(`/stories/${storyId}/run-etl`);
            
            if (response.success) {
                Utils.showAlert('ETL Agent started successfully', 'success');
                
                // Start status monitoring
                this.monitorExecution(response.executionId, storyId);
                
                // Refresh page after short delay
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                Utils.showAlert(response.message || 'Failed to start ETL Agent', 'danger');
            }
        } catch (error) {
            Utils.showAlert('Error starting ETL Agent: ' + error.message, 'danger');
        } finally {
            if (button && showProgress) {
                setTimeout(() => {
                    button.innerHTML = originalContent;
                    button.disabled = false;
                }, 1000);
            }
        }
    },

    monitorExecution: function(executionId, storyId) {
        const checkStatus = async () => {
            try {
                const status = await API.get(`/executions/${executionId}/status`);
                this.updateExecutionStatus(storyId, status);
                
                if (status.status === 'running' || status.status === 'starting') {
                    setTimeout(checkStatus, CONFIG.etlStatusCheckInterval);
                }
            } catch (error) {
                console.error('Error checking ETL status:', error);
            }
        };
        
        checkStatus();
    },

    updateExecutionStatus: function(storyId, status) {
        AppState.etlStatus[storyId] = status;
        
        // Update UI elements
        const statusElements = document.querySelectorAll(`[data-story-id="${storyId}"] .etl-status`);
        statusElements.forEach(element => {
            this.renderStatusBadge(element, status);
        });
        
        // Update progress bars if present
        const progressElements = document.querySelectorAll(`[data-story-id="${storyId}"] .etl-progress`);
        progressElements.forEach(element => {
            this.updateProgressBar(element, status);
        });
        
        // Trigger custom event
        document.dispatchEvent(new CustomEvent('etlStatusUpdate', {
            detail: { storyId, status }
        }));
    },

    renderStatusBadge: function(element, status) {
        const statusConfig = {
            'starting': { class: 'bg-info', icon: 'clock', text: 'Starting' },
            'running': { class: 'bg-primary', icon: 'spinner fa-spin', text: 'Running' },
            'completed': { class: 'bg-success', icon: 'check', text: 'Completed' },
            'failed': { class: 'bg-danger', icon: 'times', text: 'Failed' },
            'cancelled': { class: 'bg-warning', icon: 'ban', text: 'Cancelled' }
        };
        
        const config = statusConfig[status.status] || statusConfig['failed'];
        
        element.innerHTML = `
            <span class="badge ${config.class}">
                <i class="fas fa-${config.icon}"></i> ${config.text}
            </span>
        `;
    },

    updateProgressBar: function(element, status) {
        if (status.progress !== undefined) {
            const progressBar = element.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = `${status.progress}%`;
                progressBar.setAttribute('aria-valuenow', status.progress);
                
                const progressText = element.querySelector('.progress-text');
                if (progressText) {
                    progressText.textContent = `${status.progress}%`;
                }
            }
        }
    }
};

// Search and filtering functionality
const SearchFilter = {
    init: function() {
        // Initialize search inputs
        const searchInputs = document.querySelectorAll('.search-input');
        searchInputs.forEach(input => {
            input.addEventListener('input', 
                Utils.debounce(this.performSearch.bind(this), 300)
            );
        });

        // Initialize filter dropdowns
        const filterSelects = document.querySelectorAll('.filter-select');
        filterSelects.forEach(select => {
            select.addEventListener('change', this.performFilter.bind(this));
        });
    },

    performSearch: function(event) {
        const searchTerm = event.target.value.toLowerCase();
        const targetSelector = event.target.dataset.target || '.searchable-item';
        const items = document.querySelectorAll(targetSelector);
        
        items.forEach(item => {
            const searchableText = item.textContent.toLowerCase();
            const shouldShow = searchTerm === '' || searchableText.includes(searchTerm);
            item.style.display = shouldShow ? '' : 'none';
        });
        
        this.updateResultCount(items.length, 
            Array.from(items).filter(item => item.style.display !== 'none').length
        );
    },

    performFilter: function(event) {
        const filterValue = event.target.value;
        const filterType = event.target.dataset.filter;
        const targetSelector = event.target.dataset.target || '.filterable-item';
        const items = document.querySelectorAll(targetSelector);
        
        items.forEach(item => {
            const itemValue = item.dataset[filterType];
            const shouldShow = filterValue === '' || itemValue === filterValue;
            item.style.display = shouldShow ? '' : 'none';
        });
    },

    updateResultCount: function(total, visible) {
        const countElements = document.querySelectorAll('.result-count');
        countElements.forEach(element => {
            element.textContent = `Showing ${visible} of ${total} items`;
        });
    }
};

// Form validation and submission
const FormHandler = {
    init: function() {
        const forms = document.querySelectorAll('form[data-validate]');
        forms.forEach(form => {
            form.addEventListener('submit', this.handleSubmit.bind(this));
        });

        // Real-time validation
        const inputs = document.querySelectorAll('input[required], textarea[required], select[required]');
        inputs.forEach(input => {
            input.addEventListener('blur', this.validateField.bind(this));
        });
    },

    handleSubmit: function(event) {
        event.preventDefault();
        const form = event.target;
        
        if (this.validateForm(form)) {
            this.submitForm(form);
        }
    },

    validateForm: function(form) {
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });
        
        return isValid;
    },

    validateField: function(field) {
        const value = field.value.trim();
        const isValid = value.length > 0;
        
        this.setFieldValidation(field, isValid, 
            isValid ? '' : 'This field is required'
        );
        
        return isValid;
    },

    setFieldValidation: function(field, isValid, message) {
        const feedbackElement = field.parentElement.querySelector('.invalid-feedback') ||
                               this.createFeedbackElement(field);
        
        if (isValid) {
            field.classList.remove('is-invalid');
            field.classList.add('is-valid');
            feedbackElement.style.display = 'none';
        } else {
            field.classList.remove('is-valid');
            field.classList.add('is-invalid');
            feedbackElement.textContent = message;
            feedbackElement.style.display = 'block';
        }
    },

    createFeedbackElement: function(field) {
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        field.parentElement.appendChild(feedback);
        return feedback;
    },

    async submitForm(form) {
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        submitButton.disabled = true;
        
        try {
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            const response = await API.post(form.action, data);
            
            if (response.success) {
                Utils.showAlert(response.message || 'Form submitted successfully', 'success');
                
                if (response.redirect) {
                    window.location.href = response.redirect;
                } else if (form.dataset.reset === 'true') {
                    form.reset();
                }
            } else {
                Utils.showAlert(response.message || 'Form submission failed', 'danger');
            }
        } catch (error) {
            Utils.showAlert('Error submitting form: ' + error.message, 'danger');
        } finally {
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }
    }
};

// Modal management
const ModalManager = {
    init: function() {
        // Initialize modals
        const modalTriggers = document.querySelectorAll('[data-bs-toggle="modal"]');
        modalTriggers.forEach(trigger => {
            trigger.addEventListener('click', this.handleModalTrigger.bind(this));
        });
    },

    handleModalTrigger: function(event) {
        const trigger = event.currentTarget;
        const modalId = trigger.dataset.bsTarget;
        const modal = document.querySelector(modalId);
        
        if (modal && trigger.dataset.modalData) {
            this.populateModal(modal, JSON.parse(trigger.dataset.modalData));
        }
    },

    populateModal: function(modal, data) {
        Object.entries(data).forEach(([key, value]) => {
            const element = modal.querySelector(`[data-field="${key}"]`);
            if (element) {
                if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                    element.value = value;
                } else {
                    element.textContent = value;
                }
            }
        });
    }
};

// Auto-refresh functionality
const AutoRefresh = {
    start: function(interval = CONFIG.refreshInterval) {
        this.stop();
        AppState.refreshTimer = setInterval(() => {
            this.refresh();
        }, interval);
    },

    stop: function() {
        if (AppState.refreshTimer) {
            clearInterval(AppState.refreshTimer);
            AppState.refreshTimer = null;
        }
    },

    refresh: function() {
        // Check if there are any running ETL processes
        const runningStatuses = document.querySelectorAll('.etl-status .badge.bg-primary');
        
        if (runningStatuses.length > 0) {
            // Refresh ETL statuses
            this.refreshETLStatuses();
        }
        
        // Update relative timestamps
        this.updateTimestamps();
    },

    refreshETLStatuses: async function() {
        try {
            const storyIds = Array.from(document.querySelectorAll('[data-story-id]'))
                                 .map(el => el.dataset.storyId);
            
            if (storyIds.length > 0) {
                const statuses = await API.post('/api/etl-statuses', { storyIds });
                
                Object.entries(statuses).forEach(([storyId, status]) => {
                    ETLAgent.updateExecutionStatus(storyId, status);
                });
            }
        } catch (error) {
            console.error('Error refreshing ETL statuses:', error);
        }
    },

    updateTimestamps: function() {
        const timestamps = document.querySelectorAll('[data-timestamp]');
        timestamps.forEach(element => {
            const timestamp = element.dataset.timestamp;
            element.textContent = Utils.formatDateTime(timestamp);
        });
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all modules
    SearchFilter.init();
    FormHandler.init();
    ModalManager.init();
    
    // Start auto-refresh if there are ETL processes
    const hasETLProcesses = document.querySelectorAll('.etl-status').length > 0;
    if (hasETLProcesses) {
        AutoRefresh.start();
    }
    
    // Add global error handling
    window.addEventListener('error', function(event) {
        console.error('Global error:', event.error);
        Utils.showAlert('An unexpected error occurred', 'danger');
    });
    
    // Add visibility change handler to pause/resume auto-refresh
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            AutoRefresh.stop();
        } else {
            const hasETLProcesses = document.querySelectorAll('.etl-status').length > 0;
            if (hasETLProcesses) {
                AutoRefresh.start();
            }
        }
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    console.log('DevOps Interface initialized successfully');
});

// Export for global use
window.DevOpsInterface = {
    Utils,
    API,
    ETLAgent,
    SearchFilter,
    FormHandler,
    ModalManager,
    AutoRefresh,
    AppState,
    CONFIG
};