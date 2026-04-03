from flask import Flask, render_template, request, session, redirect, url_for
import random
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "aiml_project_suhani"

# --- RL PARAMETERS ---
alpha = 0.1
gamma = 0.9
epsilon = 0.6

response_map = {"bad": 0, "average": 1, "good": 2}

# --- QUESTIONS ---
questions_pool = [
    {"text": "How did you sleep last night?", "tag": "health", "options": [("bad", "Tossing and turning 😫"), ("average", "Okay 😴"), ("good", "Great ✨")]},
    {"text": "Is work/college overwhelming?", "tag": "stress", "options": [("bad", "Too much 📚"), ("average", "Manageable 😐"), ("good", "All good 💪")]},
    {"text": "Did you go outside today?", "tag": "lifestyle", "options": [("bad", "No 🏠"), ("average", "A little 🚶"), ("good", "Yes 🌳")]},
    {"text": "Energy levels?", "tag": "health", "options": [("bad", "Drained 🪫"), ("average", "Okay ☕"), ("good", "High ⚡")]},
    {"text": "Did you eat well?", "tag": "health", "options": [("bad", "Skipped 😔"), ("average", "Snack 🍎"), ("good", "Good meal 🍲")]},
    {"text": "Talked to someone?", "tag": "social", "options": [("bad", "No 😶"), ("average", "Little 📱"), ("good", "Yes ❤️")]},
    {"text": "Small win today?", "tag": "positive", "options": [("bad", "None ☁️"), ("average", "Small ✅"), ("good", "Big 🏆")]},
    {"text": "Stress level?", "tag": "stress", "options": [("bad", "High 😣"), ("average", "Okay 😐"), ("good", "Low 💪")]},
    {"text": "Feeling positive?", "tag": "emotion", "options": [("bad", "No 😔"), ("average", "Neutral 😐"), ("good", "Yes 😊")]},
    {"text": "Do you feel supported?", "tag": "social", "options": [("bad", "No 😞"), ("average", "Somewhat 🤝"), ("good", "Yes ❤️")]},
    {"text": "Did you relax?", "tag": "lifestyle", "options": [("bad", "No 😫"), ("average", "A bit 😴"), ("good", "Yes 🧘")]},
    {"text": "Focus level?", "tag": "productivity", "options": [("bad", "Poor 😵"), ("average", "Okay ⏳"), ("good", "Great 🎯")]},
    {"text": "Did you exercise?", "tag": "health", "options": [("bad", "No 🪑"), ("average", "A bit 🚶"), ("good", "Yes 💪")]},
    {"text": "Hopeful about tomorrow?", "tag": "emotion", "options": [("bad", "No ☁️"), ("average", "Maybe 🤔"), ("good", "Yes 🌟")]},
    {"text": "Ready to end positively?", "tag": "positive", "options": [("bad", "Not sure 😐"), ("average", "Okay 🙂"), ("good", "Yes ✨")]},

    # --- NEW 15 QUESTIONS ---
    {"text": "Do you feel anxious today?", "tag": "emotion", "options": [("bad", "Very 😰"), ("average", "A little 😐"), ("good", "Not at all 😊")]},
    {"text": "Did you smile today?", "tag": "positive", "options": [("bad", "No 😔"), ("average", "A bit 🙂"), ("good", "Yes 😄")]},
    {"text": "Are you feeling lonely?", "tag": "social", "options": [("bad", "Yes 😞"), ("average", "Sometimes 😐"), ("good", "No ❤️")]},
    {"text": "Did you enjoy your meals?", "tag": "health", "options": [("bad", "No 😕"), ("average", "Okay 😐"), ("good", "Yes 😋")]},
    {"text": "How productive were you?", "tag": "productivity", "options": [("bad", "Not at all 😣"), ("average", "Somewhat 😐"), ("good", "Very 💼")]},
    {"text": "Did you take breaks?", "tag": "lifestyle", "options": [("bad", "No 😫"), ("average", "Few 😐"), ("good", "Yes 🧘")]},
    {"text": "Do you feel confident today?", "tag": "emotion", "options": [("bad", "No 😔"), ("average", "A bit 😐"), ("good", "Yes 💪")]},
    {"text": "Did something upset you?", "tag": "emotion", "options": [("bad", "Yes 😢"), ("average", "A little 😐"), ("good", "No 😊")]},
    {"text": "Are you satisfied with today?", "tag": "positive", "options": [("bad", "No 😞"), ("average", "Okay 😐"), ("good", "Yes 😌")]},
    {"text": "Did you help someone today?", "tag": "social", "options": [("bad", "No 😶"), ("average", "Maybe 🤝"), ("good", "Yes ❤️")]},
    {"text": "How calm do you feel?", "tag": "emotion", "options": [("bad", "Not calm 😣"), ("average", "Okay 😐"), ("good", "Very calm 🧘")]},
    {"text": "Did you feel motivated?", "tag": "productivity", "options": [("bad", "No 😞"), ("average", "A bit 😐"), ("good", "Yes 🔥")]},
    {"text": "Did you get enough rest?", "tag": "health", "options": [("bad", "No 😴"), ("average", "Some 😐"), ("good", "Yes 🛌")]},
    {"text": "Did you enjoy your day overall?", "tag": "positive", "options": [("bad", "No 😔"), ("average", "Okay 😐"), ("good", "Yes 😄")]},
    {"text": "Do you feel mentally relaxed?", "tag": "emotion", "options": [("bad", "No 😣"), ("average", "Somewhat 😐"), ("good", "Yes 😊")]}
]

# --- CONSTANTS ---
TOTAL_QUESTIONS = len(questions_pool)
MAX_STEPS = 15

# Load or initialize Q-table
if os.path.exists("q_table.npy"):
    q_table = np.load("q_table.npy")
else:
    q_table = np.zeros((MAX_STEPS, 3, TOTAL_QUESTIONS))

# --- REWARD FUNCTION ---
def get_reward(prev_tag, current_tag, user_response):
    if prev_tag is None:
        return 0
    if user_response == "bad":
        return 2 if current_tag == prev_tag else -1
    elif user_response == "good":
        return 1
    return 0

# --- HOME ---
@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

# --- START ---
@app.route('/start', methods=['POST'])
def start():
    session['name'] = request.form.get('name')
    session['step'] = 0
    session['prev_tag'] = None
    session['moods'] = []

    # First question
    if random.random() < epsilon:
        q_idx = random.randint(0, TOTAL_QUESTIONS - 1)
    else:
        q_idx = int(np.argmax(q_table[0, 1]))

    session['current_q'] = q_idx
    session['asked'] = [q_idx]

    return redirect(url_for('chat'))

# --- CHAT ---
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global epsilon

    step = session.get('step', 0)

    if step >= MAX_STEPS:
        np.save("q_table.npy", q_table)
        
        print("Q-table saved successfully!")
        return redirect(url_for('result'))

    current_q = session.get('current_q')
    prev_tag = session.get('prev_tag')
    asked = session.get('asked', [])

    if request.method == 'POST':
        user_response = request.form.get('mood')
        response_idx = response_map[user_response]

        current_tag = questions_pool[current_q]["tag"]

        moods = session.get('moods', [])
        moods.append(user_response)
        session['moods'] = moods

        reward = get_reward(prev_tag, current_tag, user_response)
        next_step = step + 1

        if current_q not in asked:
            asked.append(current_q)
        session['asked'] = asked

        # No repetition
        available_actions = [i for i in range(TOTAL_QUESTIONS) if i not in asked]
        if not available_actions:
            available_actions = list(range(TOTAL_QUESTIONS))

        # Next question
        if random.random() < epsilon:
            next_q = random.choice(available_actions)
        else:
            next_q = max(available_actions, key=lambda x: q_table[step, response_idx, x])

        # Q-learning update
        predict = q_table[step, response_idx, current_q]

        if step >= MAX_STEPS - 1:
            target = reward
        else:
            target = reward + gamma * q_table[next_step, response_idx, next_q]

        q_table[step, response_idx, current_q] += alpha * (target - predict)

        # Update session
        session['step'] = next_step
        session['current_q'] = next_q
        session['prev_tag'] = current_tag

        # Decay epsilon
        epsilon = max(0.1, epsilon * 0.995)

        return redirect(url_for('chat'))

    q_data = questions_pool[current_q]
    return render_template('chat.html',
                           question=q_data["text"],
                           options=q_data["options"],
                           step=step + 1)

# --- RESULT ---
@app.route('/result')
def result():
    name = session.get('name', 'Friend')
    moods = session.get('moods', [])

    bad_count = moods.count("bad")
    good_count = moods.count("good")

    if bad_count > good_count:
        conclusion = f"{name}, it seems you've been going through a tough time. Take things slowly and be kind to yourself 💛"
    elif good_count > bad_count:
        conclusion = f"{name}, you're doing really well! Keep this positive energy going ✨"
    else:
        conclusion = f"{name}, you're managing things steadily. Keep going 👍"

    return render_template('result.html', name=name, conclusion=conclusion)

@app.route('/reset')
def reset():
    global q_table, epsilon

    # Reset Q-table
    q_table = np.zeros((MAX_STEPS, 3, TOTAL_QUESTIONS))

    # Delete saved file if exists
    if os.path.exists("q_table.npy"):
        os.remove("q_table.npy")

    # Reset epsilon
    epsilon = 0.6

    print("Q-table RESET successfully!")

    return redirect(url_for('index'))

@app.route('/qtable')
def show_qtable():
    import matplotlib
    matplotlib.use('Agg')  # for server (important)

    # Take average across responses (bad, avg, good)
    avg_q = np.mean(q_table, axis=1)

    plt.figure()
    plt.imshow(avg_q, aspect='auto')
    plt.colorbar()
    plt.title("Q-Table Heatmap")
    plt.xlabel("Questions")
    plt.ylabel("Steps")

    # Save image
    img_path = "static/qtable.png"
    plt.savefig(img_path)
    plt.close()

    return render_template("qtable.html", img=img_path)
# --- RUN ---
if __name__ == '__main__':
    app.run(debug=True)