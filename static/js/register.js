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
                type: "circle"
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
                bounce: false
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

    const registerForm = document.getElementById('register-form');
    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    const inputs = document.querySelectorAll('input');

    // Password visibility toggle
    togglePasswordButtons.forEach(button => {
        button.addEventListener('click', () => {
            const input = button.previousElementSibling;
            const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
            input.setAttribute('type', type);
            button.classList.toggle('active');
        });
    });

    // Input validation and visual feedback
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.parentElement.classList.add('focus');
        });

        input.addEventListener('blur', () => {
            input.parentElement.parentElement.classList.remove('focus');
            validateInput(input);
        });
    });

    function validateInput(input) {
        const inputGroup = input.parentElement.parentElement;
        inputGroup.classList.remove('error', 'success');

        if (input.value.trim() === '') return;

        if (input.type === 'email') {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            inputGroup.classList.add(emailRegex.test(input.value) ? 'success' : 'error');
        } else if (input.type === 'tel') {
            const phoneRegex = /^\d{10}$/;
            inputGroup.classList.add(phoneRegex.test(input.value) ? 'success' : 'error');
        } else if (input.type === 'password') {
            inputGroup.classList.add(input.value.length >= 6 ? 'success' : 'error');
        }
    }

    // Form submission with animation
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Validate all inputs
            inputs.forEach(validateInput);
            if (document.querySelector('.input-group.error')) return;

            const button = registerForm.querySelector('button[type="submit"]');
            button.classList.add('loading');
            button.disabled = true;

            try {
                const name = document.getElementById('register-name').value;
                const email = document.getElementById('register-email').value;
                const mobile = document.getElementById('register-mobile').value;
                const password = document.getElementById('register-password').value;
                const confirm_password = document.getElementById('register-confirm-password').value;

                const response = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, email, mobile, password, confirm_password })
                });

                const data = await response.json();

                if (response.ok) {
                    localStorage.setItem('token', data.token);
                    localStorage.setItem('name', data.username);
                    
                    // Success animation
                    button.innerHTML = '<span class="button-text">Success!</span>';
                    button.style.background = 'linear-gradient(45deg, #2ecc71, #27ae60)';
                    
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 1000);
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                button.classList.remove('loading');
                button.disabled = false;
                
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = error.message || 'Registration failed. Please try again.';
                button.parentElement.insertBefore(errorMsg, button);
                
                setTimeout(() => errorMsg.remove(), 5000);
            }
        });
    }

    // Add subtle parallax effect to the register card
    document.addEventListener('mousemove', (e) => {
        const card = document.querySelector('.register-card');
        const { clientX, clientY } = e;
        const { innerWidth, innerHeight } = window;
        
        const xAxis = (innerWidth / 2 - clientX) / 25;
        const yAxis = (innerHeight / 2 - clientY) / 25;
        
        card.style.transform = `translateY(-5px) rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    });

    // Reset transform when mouse leaves
    document.addEventListener('mouseleave', () => {
        const card = document.querySelector('.register-card');
        card.style.transform = 'translateY(0) rotateY(0) rotateX(0)';
    });
});