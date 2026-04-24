// Loading spinner
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function() {
        const btn = this.querySelector('button[type="submit"]');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        btn.disabled = true;
    });
});

// Multi-select helper
document.querySelectorAll('select[multiple]').forEach(select => {
    select.addEventListener('change', function() {
        const selected = Array.from(this.selectedOptions).map(opt => opt.value);
        console.log('Selected features:', selected);
    });
});