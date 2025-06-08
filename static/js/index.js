let currentUserId = null;
let currentChatUserId = null;
let token = null;

document.addEventListener('DOMContentLoaded', async () => {
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#ffffff" },
            shape: { type: "circle" },
            opacity: {
                value: 0.5,
                random: true,
                anim: { enable: true, speed: 1, opacity_min: 0.1, sync: false }
            },
            size: {
                value: 3,
                random: true,
                anim: { enable: true, speed: 2, size_min: 0.1, sync: false }
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
                onhover: { enable: true, mode: "grab" },
                onclick: { enable: true, mode: "push" },
                resize: true
            },
            modes: {
                grab: { distance: 140, line_linked: { opacity: 1 } },
                push: { particles_nb: 4 }
            }
        },
        retina_detect: true
    });

    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    token = localStorage.getItem('token');
    console.log('Token retrieved from localStorage:', token);
    if (!token) {
        console.log('No token found, redirecting to login');
        window.location.href = '/login';
        return;
    }

    try {
        const response = await fetch('/me', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
            const data = await response.json();
            currentUserId = data.id;
            document.getElementById('current-username').textContent = localStorage.getItem('name') || 'User';
            await loadContacts();
        } else {
            console.error('Failed to fetch user ID:', response.status, response.statusText);
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Error fetching user ID:', error);
        window.location.href = '/login';
    }

    const messageInput = document.getElementById('message-input');
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    const sendBtn = document.querySelector('.send-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    // Close context menu when clicking outside
    document.addEventListener('click', (e) => {
        const contextMenu = document.querySelector('.delete-context-menu');
        if (contextMenu && !contextMenu.contains(e.target) && !e.target.classList.contains('delete-indicator')) {
            console.log('Closing context menu due to outside click');
            contextMenu.remove();
        }
    });
});

async function loadContacts() {
    try {
        const response = await fetch('/contacts', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) throw new Error(`Failed to load contacts: ${response.statusText}`);
        
        const contacts = await response.json();
        console.log('Loaded contacts:', contacts);
        const contactList = document.getElementById('contact-list');
        contactList.innerHTML = '';
        
        contacts.forEach((contact, index) => {
            const div = document.createElement('div');
            div.className = 'contact';
            div.style.animationDelay = `${index * 0.1}s`;
            div.innerHTML = `
                <div class="avatar"></div>
                <div class="contact-info">
                    <div class="contact-name">${contact.username}</div>
                </div>
            `;
            div.onclick = () => loadChat(contact.id, contact.username);
            contactList.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading contacts:', error);
        alert(`Failed to load contacts: ${error.message}`);
    }
}

async function loadChat(userId, username) {
    console.log(`Loading chat for userId: ${userId}, username: ${username}`);
    if (currentChatUserId !== userId) {
        document.querySelector('.chat-header-info').textContent = username;
    }
    currentChatUserId = userId;

    try {
        const response = await fetch(`/chat/${userId}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) {
            if (response.status === 401) {
                console.error('Unauthorized: Invalid or expired token');
                window.location.href = '/login';
                return;
            }
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to load chat');
        }

        const messages = await response.json();
        console.log(`Fetched ${messages.length} messages for userId: ${userId}`);
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = '';

        messages.forEach((msg, index) => {
            console.log(`Rendering message: ${msg.text}, Timestamp: ${msg.timestamp}, Toxicity: ${JSON.stringify(msg.toxicity)}`);
            const div = document.createElement('div');
            div.className = `message ${msg.sender_id === currentUserId ? 'sent' : 'received'}`;
            div.style.animationDelay = `${index * 0.1}s`;
            div.dataset.messageId = msg._id; // Use MongoDB _id instead of timestamp

            const indicator = msg.toxicity.is_toxic ? '⚠️' : '✅';
            const scores = Object.entries(msg.toxicity.scores)
                .map(([cat, score]) => `${cat}: ${score.toFixed(4)}`)
                .join('<br>');
            const toxicWordsText = msg.toxicity.is_toxic && msg.toxicity.toxic_words.length > 0
                ? `Toxic words: ${msg.toxicity.toxic_words.join(', ')}`
                : 'No toxic words detected';
            const tooltipContent = `${toxicWordsText}<br>Toxicity Scores:<br>${scores}`;

            // Parse timestamp as UTC and convert to Asia/Kolkata
            const messageDate = new Date(msg.timestamp + 'Z'); // Append 'Z' to treat as UTC
            const kolkataTimeZone = 'Asia/Kolkata';
            const formattedTime = messageDate.toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true,
                timeZone: kolkataTimeZone
            });

            div.innerHTML = `
                <span class="message-text">${msg.text}</span>
                <span class="message-time">${formattedTime}</span>
                <span class="toxicity-indicator">${indicator}</span>
                <span class="delete-indicator">🗑️</span>
                <span class="tooltip">${tooltipContent}</span>
            `;

            div.addEventListener('mouseenter', () => {
                const tooltip = div.querySelector('.tooltip');
                const messageRect = div.getBoundingClientRect();
                const chatMessagesRect = chatMessages.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();

                const spaceBelow = chatMessagesRect.bottom - messageRect.bottom;
                if (spaceBelow < tooltipRect.height + 10) {
                    tooltip.classList.add('tooltip-above');
                } else {
                    tooltip.classList.remove('tooltip-above');
                }

                const spaceLeft = messageRect.left - chatMessagesRect.left;
                const tooltipHalfWidth = tooltipRect.width / 2;
                if (spaceLeft < tooltipHalfWidth) {
                    const offset = tooltipHalfWidth - spaceLeft;
                    tooltip.style.left = `${50 + (offset / messageRect.width * 100)}%`;
                    tooltip.style.transform = tooltip.classList.contains('tooltip-above') 
                        ? `translate(-50%, -100%)` 
                        : `translate(-50%, 100%)`;
                } else {
                    tooltip.style.left = '50%';
                    tooltip.style.transform = tooltip.classList.contains('tooltip-above') 
                        ? `translate(-50%, -100%)` 
                        : `translate(-50%, 100%)`;
                }

                const spaceRight = chatMessagesRect.right - messageRect.right;
                if (spaceRight < tooltipHalfWidth) {
                    const offset = tooltipHalfWidth - spaceRight;
                    tooltip.style.left = `${50 - (offset / messageRect.width * 100)}%`;
                    tooltip.style.transform = tooltip.classList.contains('tooltip-above') 
                        ? `translate(-50%, -100%)` 
                        : `translate(-50%, 100%)`;
                }
            });

            // Add delete functionality using event delegation
            div.addEventListener('click', (e) => {
                if (e.target.classList.contains('delete-indicator')) {
                    e.stopPropagation();
                    e.preventDefault();
                    console.log('Delete indicator clicked for message:', msg.text);
                    showDeleteContextMenu(div, msg.sender_id === currentUserId);
                }
            });

            chatMessages.appendChild(div);
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Error loading chat:', error);
        alert(`Failed to load chat: ${error.message}`);
    }
}

function showDeleteContextMenu(messageDiv, isSender) {
    console.log('Showing delete context menu');
    // Remove any existing context menu
    const existingMenu = document.querySelector('.delete-context-menu');
    if (existingMenu) {
        existingMenu.remove();
    }

    const contextMenu = document.createElement('div');
    contextMenu.className = 'delete-context-menu';
    // Ensure visibility
    contextMenu.style.background = 'var(--bg-primary, #ffffff)';
    contextMenu.style.border = 'var(--border-color, 1px solid #ccc)';
    contextMenu.style.color = 'var(--text-primary, #000000)';
    contextMenu.style.zIndex = '1000';
    contextMenu.style.display = 'block';

    // "Delete for Me" option
    const deleteForMeOption = document.createElement('div');
    deleteForMeOption.className = 'delete-option';
    deleteForMeOption.textContent = 'Delete for Me';
    deleteForMeOption.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log('Delete for Me clicked');
        deleteMessage(messageDiv.dataset.messageId, 'me');
        contextMenu.remove();
    });
    contextMenu.appendChild(deleteForMeOption);

    // "Delete for Everyone" option (only for sender)
    if (isSender) {
        const deleteForEveryoneOption = document.createElement('div');
        deleteForEveryoneOption.className = 'delete-option';
        deleteForEveryoneOption.textContent = 'Delete for Everyone';
        deleteForEveryoneOption.addEventListener('click', (e) => {
            e.stopPropagation();
            console.log('Delete for Everyone clicked');
            deleteMessage(messageDiv.dataset.messageId, 'everyone');
            contextMenu.remove();
        });
        contextMenu.appendChild(deleteForEveryoneOption);
    }

    // Position below the message
    const messageRect = messageDiv.getBoundingClientRect();
    const chatMessages = document.getElementById('chat-messages');
    const chatMessagesRect = chatMessages.getBoundingClientRect();

    contextMenu.style.position = 'absolute';
    contextMenu.style.left = `${messageRect.left - chatMessagesRect.left}px`;
    contextMenu.style.top = `${messageRect.bottom - chatMessagesRect.top + chatMessages.scrollTop + 5}px`;

    chatMessages.appendChild(contextMenu);

    // Adjust if out of bounds
    const contextMenuRect = contextMenu.getBoundingClientRect();
    if (contextMenuRect.right > chatMessagesRect.right) {
        contextMenu.style.left = `${chatMessagesRect.right - chatMessagesRect.left - contextMenuRect.width - 5}px`;
    }
    if (contextMenuRect.bottom > chatMessagesRect.bottom) {
        contextMenu.style.top = `${messageRect.top - chatMessagesRect.top + chatMessages.scrollTop - contextMenuRect.height - 5}px`;
    }

    console.log('Context menu positioned at:', contextMenu.style.left, contextMenu.style.top);
}

async function deleteMessage(messageId, option) {
    try {
        console.log(`Deleting message ${messageId} with option: ${option}`);
        const response = await fetch(`/chat/${currentChatUserId}/message/${messageId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ option })
        });

        if (response.ok) {
            console.log(`Message ${messageId} deleted with option: ${option}`);
            await loadChat(currentChatUserId, document.querySelector('.chat-header-info').textContent);
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to delete message');
        }
    } catch (error) {
        console.error('Error deleting message:', error);
        alert(`Failed to delete message: ${error.message}`);
    }
}

async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const text = messageInput.value.trim();
    
    if (!text || !currentChatUserId) {
        if (!text) {
            messageInput.classList.add('shake');
            setTimeout(() => messageInput.classList.remove('shake'), 500);
        }
        return;
    }

    const chatMessages = document.getElementById('chat-messages');
    const tempDiv = document.createElement('div');
    tempDiv.className = 'message sent';
    tempDiv.style.animationDelay = '0s';
    tempDiv.dataset.messageId = `temp-${Date.now()}`;

    // Use current time in Asia/Kolkata for temporary message
    const now = new Date();
    const kolkataTimeZone = 'Asia/Kolkata';
    const formattedTime = now.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
        timeZone: kolkataTimeZone
    });

    tempDiv.innerHTML = `
        <span class="message-text">${text}</span>
        <span class="message-time">${formattedTime}</span>
        <span class="toxicity-indicator">⏳</span>
        <span class="tooltip">Processing...</span>
    `;
    chatMessages.appendChild(tempDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        console.log('Sending message:', text);
        const response = await fetch(`/chat/${currentChatUserId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const responseData = await response.json();
            console.log('Message sent successfully:', responseData);
            messageInput.value = '';
            await new Promise(resolve => setTimeout(resolve, 200));
            await loadChat(currentChatUserId, document.querySelector('.chat-header-info').textContent);
        } else {
            if (response.status === 401) {
                console.error('Unauthorized: Invalid or expired token');
                window.location.href = '/login';
                return;
            }
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to send message');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        alert(`Failed to send message: ${error.message}`);
        tempDiv.remove();
    }
}

function refreshChat() {
    const refreshBtn = document.querySelector('.refresh-btn');
    refreshBtn.style.transform = 'rotate(360deg)';
    
    if (currentChatUserId) {
        loadChat(currentChatUserId, document.querySelector('.chat-header-info').textContent);
    }
    
    setTimeout(() => {
        refreshBtn.style.transform = 'rotate(0deg)';
    }, 500);
}

function logout() {
    const logoutBtn = document.querySelector('.logout-btn');
    logoutBtn.style.transform = 'rotate(360deg)';
    
    setTimeout(() => {
        localStorage.removeItem('token');
        localStorage.removeItem('name');
        window.location.href = '/login';
    }, 500);
}