document.addEventListener('DOMContentLoaded', () => {
    // Initialize particles.js
    particlesJS('particles-js', {
        particles: {
            number: {
                value: 80,
                density: {
                    enable: true,
                    value_area: 800
                }
            },
            color: {
                value: "#ffffff"
            },
            shape: {
                type: "circle",
                stroke: {
                    width: 0,
                    color: "#000000"
                },
            },
            opacity: {
                value: 0.5,
                random: true,
                anim: {
                    enable: true,
                    speed: 1,
                    opacity_min: 0.1,
                    sync: false
                }
            },
            size: {
                value: 3,
                random: true,
                anim: {
                    enable: true,
                    speed: 2,
                    size_min: 0.1,
                    sync: false
                }
            },
            line_linked: {
                enable: true,
                distance: 150,
                color: "#ffffff",
                opacity: 0.4,
                width: 1
            },
            move: {
                enable: true,
                speed: 1,
                direction: "none",
                random: true,
                straight: false,
                out_mode: "out",
                bounce: false,
            }
        },
        interactivity: {
            detect_on: "canvas",
            events: {
                onhover: {
                    enable: true,
                    mode: "grab"
                },
                onclick: {
                    enable: true,
                    mode: "push"
                },
                resize: true
            },
            modes: {
                grab: {
                    distance: 140,
                    line_linked: {
                        opacity: 1
                    }
                },
                push: {
                    particles_nb: 4
                }
            }
        },
        retina_detect: true
    });

    // Form elements
    const loginForm = document.getElementById('login-form');
    const emailInput = document.getElementById('login-email');
    const passwordInput = document.getElementById('login-password');
    const togglePasswordBtn = document.getElementById('toggle-password');
    const loginButton = document.querySelector('.login-button');

    // Forgot Password elements
    const forgotPasswordLink = document.getElementById('forgot-password-link');
    const forgotPasswordModal = document.getElementById('forgot-password-modal');
    const closeForgotModal = document.getElementById('close-forgot-modal');
    const forgotPasswordForm = document.getElementById('forgot-password-form');
    const forgotEmailInput = document.getElementById('forgot-email');
    const newPasswordInput = document.getElementById('new-password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const toggleNewPasswordBtn = document.getElementById('toggle-new-password');
    const toggleConfirmPasswordBtn = document.getElementById('toggle-confirm-password');
    const updateButton = document.querySelector('.update-button');

    // Toggle password visibility for login form
    if (togglePasswordBtn) {
        togglePasswordBtn.addEventListener('click', () => {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            
            // Change the eye icon
            const eyeIcon = togglePasswordBtn.querySelector('.eye-icon');
            if (type === 'text') {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle><path d="m3 3 18 18"></path>';
            } else {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle>';
            }
        });
    }

    // Toggle password visibility for new password
    if (toggleNewPasswordBtn) {
        toggleNewPasswordBtn.addEventListener('click', () => {
            const type = newPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            newPasswordInput.setAttribute('type', type);
            const eyeIcon = toggleNewPasswordBtn.querySelector('.eye-icon');
            if (type === 'text') {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle><path d="m3 3 18 18"></path>';
            } else {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle>';
            }
        });
    }

    // Toggle password visibility for confirm password
    if (toggleConfirmPasswordBtn) {
        toggleConfirmPasswordBtn.addEventListener('click', () => {
            const type = confirmPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            confirmPasswordInput.setAttribute('type', type);
            const eyeIcon = toggleConfirmPasswordBtn.querySelector('.eye-icon');
            if (type === 'text') {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle><path d="m3 3 18 18"></path>';
            } else {
                eyeIcon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle>';
            }
        });
    }

    // Show forgot password modal
    if (forgotPasswordLink) {
        forgotPasswordLink.addEventListener('click', (e) => {
            e.preventDefault();
            forgotPasswordModal.style.display = 'flex';
            setTimeout(() => forgotPasswordModal.classList.add('active'), 10);
            // Pre-fill email if available
            if (emailInput.value) {
                forgotEmailInput.value = emailInput.value;
                validateInput(forgotEmailInput);
            }
        });
    }

    // Close forgot password modal
    if (closeForgotModal) {
        closeForgotModal.addEventListener('click', () => {
            forgotPasswordModal.classList.remove('active');
            setTimeout(() => {
                forgotPasswordModal.style.display = 'none';
                // Clear form
                forgotPasswordForm.reset();
                const inputs = forgotPasswordForm.querySelectorAll('input');
                inputs.forEach(input => {
                    const inputGroup = input.parentElement.parentElement;
                    inputGroup.classList.remove('error', 'success');
                    const errorMsg = inputGroup.querySelector('.error-message');
                    if (errorMsg) errorMsg.remove();
                });
            }, 300);
        });
    }

    // Input focus effects for both forms
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.parentElement.classList.add('focus');
        });
        
        input.addEventListener('blur', () => {
            input.parentElement.parentElement.classList.remove('focus');
            validateInput(input);
        });
    });

    // Simple input validation
    function validateInput(input) {
        const inputGroup = input.parentElement.parentElement;
        
        // Remove any existing error message
        const existingError = inputGroup.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        inputGroup.classList.remove('error', 'success');
        
        if (input.value.trim() === '') {
            // Empty input
            return;
        }
        
        if (input.type === 'email') {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(input.value)) {
                inputGroup.classList.add('error');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Please enter a valid email address';
                inputGroup.appendChild(errorMsg);
            } else {
                inputGroup.classList.add('success');
            }
        } else if (input.type === 'password' || input.type === 'text') {
            if (input.value.length < 6) {
                inputGroup.classList.add('error');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Password must be at least 6 characters';
                inputGroup.appendChild(errorMsg);
            } else {
                inputGroup.classList.add('success');
            }
        }
    }

    // Form submission with animation for login
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Validate all inputs before submission
            inputs.forEach(validateInput);
            
            // Check if there are any errors
            const hasErrors = document.querySelector('.input-group.error');
            if (hasErrors) {
                return;
            }
            
            // Get form values
            const email = emailInput.value;
            const password = passwordInput.value;
            
            // Show loading state
            loginButton.classList.add('loading');
            
            try {
                // Existing login logic
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Success - store token and username as in original code
                    localStorage.setItem('token', data.token);
                    localStorage.setItem('name', data.username);
                    
                    // Add success animation before redirecting
                    loginButton.classList.remove('loading');
                    loginButton.innerHTML = '<span class="button-text">Success!</span>';
                    loginButton.style.background = 'linear-gradient(45deg, #2ecc71, #27ae60)';
                    
                    // Redirect after a short delay to show the success state
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 800);
                } else {
                    // Error handling with animation
                    loginButton.classList.remove('loading');
                    
                    // Show error message
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'error-message';
                    errorMsg.textContent = data.error || 'Login failed. Please check your credentials.';
                    loginForm.appendChild(errorMsg);
                    
                    // Shake the form
                    loginForm.classList.add('error');
                    setTimeout(() => {
                        loginForm.classList.remove('error');
                    }, 500);
                }
            } catch (error) {
                // Network or other error
                loginButton.classList.remove('loading');
                
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Connection error. Please try again later.';
                loginForm.appendChild(errorMsg);
            }
        });
    }

    // Form submission for forgot password
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Validate all inputs
            const forgotInputs = [forgotEmailInput, newPasswordInput, confirmPasswordInput];
            forgotInputs.forEach(validateInput);

            // Check for errors
            const hasErrors = forgotPasswordForm.querySelector('.input-group.error');
            if (hasErrors) return;

            // Additional validation: check if passwords match
            if (newPasswordInput.value !== confirmPasswordInput.value) {
                const confirmGroup = confirmPasswordInput.parentElement.parentElement;
                confirmGroup.classList.add('error');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Passwords do not match';
                confirmGroup.appendChild(errorMsg);
                return;
            }

            // Get form values
            const email = forgotEmailInput.value;
            const new_password = newPasswordInput.value;

            // Show loading state
            updateButton.classList.add('loading');

            try {
                const response = await fetch('/forgot-password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, new_password })
                });

                const data = await response.json();

                if (response.ok) {
                    // Success
                    updateButton.classList.remove('loading');
                    updateButton.innerHTML = '<span class="button-text">Success!</span>';
                    updateButton.style.background = 'linear-gradient(45deg, #2ecc71, #27ae60)';
                    setTimeout(() => {
                        forgotPasswordModal.classList.remove('active');
                        setTimeout(() => {
                            forgotPasswordModal.style.display = 'none';
                            forgotPasswordForm.reset();
                            forgotInputs.forEach(input => {
                                const inputGroup = input.parentElement.parentElement;
                                inputGroup.classList.remove('error', 'success');
                                const errorMsg = inputGroup.querySelector('.error-message');
                                if (errorMsg) errorMsg.remove();
                            });
                        }, 300);
                    }, 800);
                } else {
                    // Error
                    updateButton.classList.remove('loading');
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'error-message';
                    errorMsg.textContent = data.error || 'Failed to reset password.';
                    forgotPasswordForm.appendChild(errorMsg);
                    forgotPasswordForm.classList.add('error');
                    setTimeout(() => {
                        forgotPasswordForm.classList.remove('error');
                    }, 500);
                }
            } catch (error) {
                updateButton.classList.remove('loading');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Connection error. Please try again later.';
                forgotPasswordForm.appendChild(errorMsg);
            }
        });
    }

    // Add subtle parallax effect to the login card
    document.addEventListener('mousemove', (e) => {
        const loginCard = document.querySelector('.login-card');
        const { clientX, clientY } = e;
        const { innerWidth, innerHeight } = window;
        
        const xAxis = (innerWidth / 2 - clientX) / 25;
        const yAxis = (innerHeight / 2 - clientY) / 25;
        
        loginCard.style.transform = `translateY(-5px) rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    });
    
    // Reset transform when mouse leaves
    document.addEventListener('mouseleave', () => {
        const loginCard = document.querySelector('.login-card');
        loginCard.style.transform = 'translateY(0) rotateY(0) rotateX(0)';
    });
});