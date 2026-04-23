document.addEventListener("DOMContentLoaded", () => {
  const config = window.CHAT_CONFIG || {};
  const chatArea = document.getElementById("chat-area");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const teachBtn = document.getElementById("teach-btn");
  const teachModal = document.getElementById("teach-modal");
  const teachClose = document.getElementById("teach-close");
  const teachCancel = document.getElementById("teach-cancel");
  const teachSave = document.getElementById("teach-save");
  const teachQuestion = document.getElementById("teach-question");
  const teachIntent = document.getElementById("teach-intent");
  const teachAnswer = document.getElementById("teach-answer");
  const statusLine = document.getElementById("status-line");
  const quickButtons = document.querySelectorAll(".quick-btn");

  let waitingForResponse = false;
  let thinkingBubble = null;
  let thinkingTimer = null;
  let thinkingStep = 0;

  function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function addMessage(sender, text, role, timestamp, className = "") {
    const message = document.createElement("div");
    message.className = `message ${role} ${className}`.trim();

    const senderEl = document.createElement("div");
    senderEl.className = "sender";
    senderEl.textContent = sender;

    const timeEl = document.createElement("div");
    timeEl.className = "timestamp";
    timeEl.textContent = timestamp || new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    message.appendChild(senderEl);
    message.appendChild(timeEl);
    message.appendChild(bubble);
    chatArea.appendChild(message);
    scrollToBottom();

    return { message, bubble };
  }

  function renderWelcome() {
    const welcome = config.welcomeMessage || "Hello! How can I help you today?";
    addMessage("Counselor", welcome, "bot");
  }

  function clearChat() {
    stopThinking();
    chatArea.innerHTML = "";
    renderWelcome();
    waitingForResponse = false;
    thinkingBubble = null;
    if (statusLine) {
      statusLine.textContent = "Ready";
    }
  }

  function stopThinking() {
    if (thinkingTimer) {
      clearInterval(thinkingTimer);
      thinkingTimer = null;
    }
    thinkingStep = 0;
  }

  function startThinking() {
    const { bubble } = addMessage("Counselor", "Thinking", "bot thinking");
    thinkingBubble = bubble;

    const updateThinking = () => {
      if (!thinkingBubble) {
        return;
      }

      thinkingStep = (thinkingStep + 1) % 4;
      thinkingBubble.textContent = `Thinking${".".repeat(thinkingStep)}`;
    };

    updateThinking();
    thinkingTimer = setInterval(updateThinking, 350);
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || waitingForResponse) {
      return;
    }

    waitingForResponse = true;
    addMessage("You", text, "user");
    input.value = "";
    startThinking();
    if (statusLine) {
      statusLine.textContent = "Thinking...";
    }

    try {
      const response = await fetch(config.chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await response.json();
      stopThinking();

      if (!response.ok || !data.ok) {
        if (thinkingBubble) {
          thinkingBubble.textContent = data.error || "I could not process that request.";
          thinkingBubble.parentElement.classList.remove("thinking");
        } else {
          addMessage("Counselor", data.error || "I could not process that request.", "bot");
        }
        if (statusLine) {
          statusLine.textContent = "Ready";
        }
      } else if (thinkingBubble) {
        thinkingBubble.textContent = data.response;
        thinkingBubble.parentElement.classList.remove("thinking");
        if (statusLine) {
          statusLine.textContent = `Intent: ${data.intent} | Source: ${data.source} | Confidence: ${Number(data.confidence || 0).toFixed(2)}`;
        }
      } else {
        addMessage("Counselor", data.response, "bot");
        if (statusLine) {
          statusLine.textContent = `Intent: ${data.intent} | Source: ${data.source} | Confidence: ${Number(data.confidence || 0).toFixed(2)}`;
        }
      }
      thinkingBubble = null;
    } catch (error) {
      stopThinking();
      if (thinkingBubble) {
        thinkingBubble.textContent = "I could not reach the server right now.";
        thinkingBubble.parentElement.classList.remove("thinking");
      } else {
        addMessage("Counselor", "I could not reach the server right now.", "bot");
      }
      thinkingBubble = null;
      if (statusLine) {
        statusLine.textContent = "Ready";
      }
    } finally {
      waitingForResponse = false;
      scrollToBottom();
    }
  }

  function openTeachModal() {
    teachModal.classList.remove("hidden");
    teachQuestion.focus();
  }

  function closeTeachModal() {
    teachModal.classList.add("hidden");
    teachQuestion.value = "";
    teachIntent.value = "";
    teachAnswer.value = "";
  }

  async function saveTeachExample() {
    const question = teachQuestion.value.trim();
    const intent = teachIntent.value.trim();
    const answer = teachAnswer.value.trim();

    if (!question || !intent || !answer) {
      alert("Please fill in question, intent, and answer.");
      return;
    }

    try {
      const response = await fetch(config.teachUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, intent, answer }),
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        alert(data.error || "Could not save the example.");
        return;
      }

      closeTeachModal();
      addMessage("Counselor", "I learned that example and refreshed my model. Try asking it again.", "bot");
      if (statusLine) {
        statusLine.textContent = "New knowledge added";
      }
    } catch (error) {
      alert("Could not save the example right now.");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  });

  clearBtn.addEventListener("click", clearChat);
  teachBtn.addEventListener("click", openTeachModal);
  teachClose.addEventListener("click", closeTeachModal);
  teachCancel.addEventListener("click", closeTeachModal);
  teachSave.addEventListener("click", saveTeachExample);

  teachModal.addEventListener("click", (event) => {
    if (event.target === teachModal) {
      closeTeachModal();
    }
  });

  quickButtons.forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.question || "";
      sendMessage();
    });
  });

  renderWelcome();
});
