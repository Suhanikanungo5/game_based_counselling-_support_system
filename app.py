from flask import Flask, render_template, request, session, redirect, url_for
import random
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "aiml_project_suhani"

alpha = 0.1
gamma = 0.9
epsilon = 0.6

# --- QUESTIONS & INTERVENTIONS ---
questions_pool = [
    # --- YOUR 30 ORIGINAL ASSESSMENT QUESTIONS ---
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
    {"text": "Do you feel mentally relaxed?", "tag": "emotion", "options": [("bad", "No 😣"), ("average", "Somewhat 😐"), ("good", "Yes 😊")]},

    # --- NEW: ACTIONABLE SUGGESTIONS / INTERVENTIONS ---
    {"text": "💡 I have a suggestion: Let's pause. Take a slow, deep breath in for 4 seconds, and out for 4 seconds.", "tag": "intervention", "options": [("bad", "Didn't help 😣"), ("average", "A bit better 😐"), ("good", "I feel calmer 🧘")]},
    {"text": "💡 Quick physical check: Drop your shoulders away from your ears and unclench your jaw. Notice any difference?", "tag": "intervention", "options": [("bad", "Still tense 😫"), ("average", "Slightly better 😐"), ("good", "Much more relaxed 💪")]},
    {"text": "💡 Try drinking a glass of water right now. Dehydration secretly spikes stress levels.", "tag": "intervention", "options": [("bad", "Can't right now 🪑"), ("average", "Had a sip 💧"), ("good", "Drank a full glass 🚰")]},
    {"text": "💡 Shift your focus: Can you name one small thing that went right today, no matter how tiny?", "tag": "intervention", "options": [("bad", "Nothing went right ☁️"), ("average", "I guess one thing ✅"), ("good", "Yes, I can! 🏆")]},
    {"text": "💡 Stand up and do a quick 30-second stretch. Getting blood flowing can reset your brain.", "tag": "intervention", "options": [("bad", "Too tired 😫"), ("average", "Did a quick stretch 🚶"), ("good", "Feeling refreshed 🤸")]},
    {"text": "💡 Reminder: You are doing your best, and that is enough for today. Be kind to yourself.", "tag": "intervention", "options": [("bad", "Hard to believe 😔"), ("average", "Thanks 😐"), ("good", "I needed to hear that 🌟")]}
]

TOTAL_QUESTIONS = len(questions_pool)
MAX_STEPS = 15
MAX_MOOD = 10

TOTAL_QUESTIONS = len(questions_pool)
MAX_STEPS = 15
MAX_MOOD = 10

if os.path.exists("q_table.npy"):
    q_table = np.load("q_table.npy")
    if q_table.shape != (MAX_MOOD + 1, TOTAL_QUESTIONS):
        q_table = np.zeros((MAX_MOOD + 1, TOTAL_QUESTIONS))
else:
    q_table = np.zeros((MAX_MOOD + 1, TOTAL_QUESTIONS))

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    session['name'] = request.form.get('name')
    session['step'] = 0
    session['moods'] = []
    
    # NEW: Capture the user's starting mood from the HTML slider
    # We use a try/except block as a failsafe to default to 5 if the data is missing
    try:
        user_starting_mood = int(request.form.get('initial_mood', 5))
    except ValueError:
        user_starting_mood = 5
        
    session['mood_score'] = user_starting_mood
    
    # NEW: The AI now uses the USER'S specific mood to look up the best first question
    if random.random() < epsilon:
        q_idx = random.randint(0, TOTAL_QUESTIONS - 1)
    else:
        # It looks at the specific row for their starting mood!
        q_idx = int(np.argmax(q_table[user_starting_mood]))
        
    session['current_q'] = q_idx
    session['asked'] = [q_idx]
    
    session['user_points'] = 0
    session['ai_feedback'] = "I am ready when you are. Let's begin!"
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global epsilon
    step = session.get('step', 0)
    
    if step >= MAX_STEPS:
        np.save("q_table.npy", q_table)
        print("Q-table saved successfully!")
        return redirect(url_for('result'))
        
    current_q = session.get('current_q')
    old_mood = session.get('mood_score', 5)
    asked = session.get('asked', [])
    
    if request.method == 'POST':
        user_response = request.form.get('mood')
        
        moods = session.get('moods', [])
        moods.append(user_response)
        session['moods'] = moods
        
        if user_response == "bad":
            new_mood = max(0, old_mood - 1)
        elif user_response == "good":
            new_mood = min(10, old_mood + 1)
        else:
            new_mood = old_mood
            
        reward = new_mood - old_mood
        # --- NEW: GAMIFICATION & FEEDBACK LOGIC ---
        user_points = session.get('user_points', 0)
        
        if reward > 0:
            points_earned = reward * 10
            user_points += points_earned
            session['ai_feedback'] = f"🌟 Great job! Your mood improved. +{points_earned} Resilience Points!"
        elif reward < 0:
            session['ai_feedback'] = "💙 I hear you. It is okay to feel that way. Let's try something else."
        else:
            session['ai_feedback'] = "Got it. Thank you for sharing."
            
        session['user_points'] = user_points
        # ------------------------------------------
        next_step = step + 1
        
        if current_q not in asked:
            asked.append(current_q)
            session['asked'] = asked
            
        available_actions = [i for i in range(TOTAL_QUESTIONS) if i not in asked]
        if not available_actions:
            available_actions = list(range(TOTAL_QUESTIONS))
            
        if random.random() < epsilon:
            next_q = random.choice(available_actions)
        else:
            next_q = max(available_actions, key=lambda x: q_table[new_mood, x])
            
        predict = q_table[old_mood, current_q]
        if step >= MAX_STEPS - 1:
            target = reward
        else:
            target = reward + gamma * np.max(q_table[new_mood])
            
        q_table[old_mood, current_q] += alpha * (target - predict)
        
        session['step'] = next_step
        session['current_q'] = next_q
        session['mood_score'] = new_mood
        
        epsilon = max(0.1, epsilon * 0.995)
        
        return redirect(url_for('chat'))
        
    q_data = questions_pool[current_q]
    
    # THIS IS THE LINE THAT WAS UPDATED FOR STEP 2
    return render_template('chat.html', 
                           question=q_data["text"], 
                           options=q_data["options"], 
                           step=step + 1, 
                           mood_score=old_mood,
                           user_points=session.get('user_points', 0), # NEW
                           ai_feedback=session.get('ai_feedback', '')) # NEW
@app.route('/result')
def result():
    name = session.get('name', 'Friend')
    final_mood = session.get('mood_score', 5)
    
    if final_mood <= 3:
        heading = "Take Care 💙"
        conclusion = f"{name}, your responses indicate you are feeling quite overwhelmed right now. Please remember it's okay to take a step back, rest, and seek support from friends or a professional. 💛"
    elif final_mood >= 7:
        heading = f"Great Job, {name}! ✨"
        conclusion = f"{name}, you finished the session in a great headspace! Keep up the healthy habits that are making you feel this positive. ✨"
    else:
        heading = "Session Complete 🌟"
        conclusion = f"{name}, you are managing things steadily. Keep going 👍"
        
    return render_template('result.html', name=name, conclusion=conclusion, heading=heading)

@app.route('/reset')
def reset():
    global q_table, epsilon
    q_table = np.zeros((MAX_MOOD + 1, TOTAL_QUESTIONS))
    if os.path.exists("q_table.npy"):
        os.remove("q_table.npy")
    epsilon = 0.6
    print("Q-table RESET successfully!")
    return redirect(url_for('index'))

# --- PRE-TRAIN THE AI (SIMULATED USER) ---
@app.route('/pretrain')
def pretrain():
    global q_table, epsilon
    EPISODES = 5000 
    
    rewards_all_episodes = [] 

    for episode in range(EPISODES):
        sim_mood = random.randint(2, 8) 
        asked = []
        episode_reward = 0 

        for step in range(MAX_STEPS):
            available_actions = [i for i in range(TOTAL_QUESTIONS) if i not in asked]
            if not available_actions:
                break
                
            if random.random() < 0.2: 
                action = random.choice(available_actions)
            else:
                action = max(available_actions, key=lambda x: q_table[sim_mood, x])

            asked.append(action)
            tag = questions_pool[action]["tag"]
            old_mood = sim_mood

            if tag == "intervention":
                sim_mood += 2 if random.random() < 0.85 else 0
            elif tag in ["lifestyle", "positive"] and sim_mood <= 5:
                sim_mood += 1 if random.random() < 0.7 else 0
            elif tag in ["stress", "productivity"] and sim_mood <= 4:
                sim_mood -= 1 if random.random() < 0.8 else 0
            elif tag in ["health", "social"]:
                sim_mood += 1 if random.random() < 0.5 else 0
            else:
                sim_mood += random.choice([-1, 0, 0, 1])

            sim_mood = max(0, min(10, sim_mood))

            reward = sim_mood - old_mood
            episode_reward += reward 
            
            predict = q_table[old_mood, action]
            target = reward + gamma * np.max(q_table[sim_mood])
            q_table[old_mood, action] += alpha * (target - predict)

        rewards_all_episodes.append(episode_reward)

    np.save("q_table.npy", q_table)
    epsilon = 0.1 

    import matplotlib
    matplotlib.use('Agg')
    plt.figure(figsize=(10, 5))
    
    window = 100
    smoothed_rewards = [np.mean(rewards_all_episodes[max(0, i-window):i+1]) for i in range(len(rewards_all_episodes))]
    
    plt.plot(smoothed_rewards, color='#2193b0', linewidth=2)
    plt.title('AI Learning Curve')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_path = "static/learning_curve.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    return f"""
    <div style="font-family: sans-serif; text-align: center; margin-top: 40px; background-color: #f9f9f9; padding: 30px; border-radius: 15px; max-width: 800px; margin-left: auto; margin-right: auto;">
        <h1 style="color: #4CAF50;">Training Complete! 🧠</h1>
        <p>The AI just practiced 5,000 sessions with a simulated user.</p>
        <div style="margin: 30px 0;">
            <img src="/{img_path}" style="max-width: 100%; border-radius: 10px; border: 1px solid #ddd;">
        </div>
        <div style="margin-top: 30px;">
            <a href="/" style="padding: 12px 25px; background-color: #2193b0; color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">Go Back Home to Test It</a>
        </div>
    </div>
    """

@app.route('/qtable')
def show_qtable():
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure()
    plt.imshow(q_table, aspect='auto')
    plt.colorbar()
    plt.title("Q-Table Heatmap")
    plt.xlabel("Questions")
    plt.ylabel("Mood State")
    
    img_path = "static/qtable.png"
    plt.savefig(img_path)
    plt.close()
    
    return render_template("qtable.html", img=img_path)

if __name__ == '__main__':
    app.run(debug=True)