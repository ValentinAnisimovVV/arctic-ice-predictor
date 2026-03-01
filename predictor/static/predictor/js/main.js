// predictor/static/predictor/js/main.js

// Глобальные функции для всего приложения
const ArcticApp = {
    // Инициализация при загрузке страницы
    init: function() {
        this.initTooltips();
        this.initRangeInputs();
        this.initCopyButtons();
        this.initAutoRefresh();
    },

    // Инициализация подсказок Bootstrap
    initTooltips: function() {
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        if (tooltips.length) {
            tooltips.forEach(el => new bootstrap.Tooltip(el));
        }
    },

    // Инициализация ползунков с отображением значения
    initRangeInputs: function() {
        document.querySelectorAll('input[type=range]').forEach(range => {
            const output = range.nextElementSibling;
            if (output && output.tagName === 'OUTPUT') {
                range.addEventListener('input', function() {
                    output.value = this.value;
                });
            }
        });
    },

    // Кнопки копирования текста
    initCopyButtons: function() {
        document.querySelectorAll('[data-copy]').forEach(btn => {
            btn.addEventListener('click', function(e) {
                const text = this.getAttribute('data-copy');
                navigator.clipboard.writeText(text).then(() => {
                    this.classList.add('btn-success');
                    setTimeout(() => {
                        this.classList.remove('btn-success');
                    }, 2000);
                });
            });
        });
    },

    // Автообновление страниц с processing статусом
    initAutoRefresh: function() {
        const statusElement = document.querySelector('[data-status="processing"]');
        if (statusElement) {
            setTimeout(() => {
                window.location.reload();
            }, 5000);
        }
    },

    // Функция для загрузки данных через AJAX
    fetchPrediction: function(jobId) {
        fetch(`/api/predict/${jobId}/`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    window.location.reload();
                }
            })
            .catch(error => console.error('Error:', error));
    },

    // Функция для отображения уведомлений
    showNotification: function(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);

            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
    },

    // Форматирование чисел
    formatNumber: function(num, decimals = 2) {
        return Number(num).toFixed(decimals);
    },

    // Получение цвета в зависимости от концентрации льда
    getIceColor: function(concentration) {
        const colors = [
            '#08306b', '#08519c', '#2171b5', '#4292c6',
            '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff'
        ];
        const index = Math.min(Math.floor(concentration * colors.length), colors.length - 1);
        return colors[index];
    }
};

// Инициализация при загрузке DOM
document.addEventListener('DOMContentLoaded', function() {
    ArcticApp.init();
});

// Экспорт для использования в консоли
window.ArcticApp = ArcticApp;