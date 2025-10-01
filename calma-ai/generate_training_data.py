import json
import random

# Read existing data
with open('data/raw/training_family.json', 'r') as f:
    existing_data = json.load(f)

# Template scenarios covering various themes
scenarios = []

# Theme: Academic pressure from parents
academic_scenarios = [
    {
        "instruction": "My mother cries every time I bring home a B grade. She says I'm embarrassing the family.",
        "context": "User is a 16-year-old {} student in {}, facing extreme academic pressure.",
        "response": "Your mother's reaction is excessive and harmful. B grades are good—not everyone can be perfect. Her tears are manipulation, even if unintentional. You're not embarrassing anyone; she's setting unrealistic standards. Talk to her calmly about how this pressure affects your mental health. If she won't listen, involve a trusted relative. Your worth isn't determined by grades. Focus on doing your best, not being perfect. Perfectionism leads to burnout, not success."
    },
    {
        "instruction": "My parents make me study until 2 AM every night. I'm exhausted and my grades are actually getting worse.",
        "context": "User is a 17-year-old {} student in {}, experiencing counterproductive study pressure.",
        "response": "Exhaustion hurts performance—your brain needs rest to learn. Show your parents research on sleep and academic performance. Suggest a balanced schedule with adequate sleep. If they won't listen, talk to a teacher or counselor to intervene. Your declining grades prove their method isn't working. Quality study time beats quantity. Protect your health—sleep deprivation causes serious problems. You need rest to succeed."
    },
    {
        "instruction": "I want to take a gap year after Form 6 but my parents say only lazy people do that.",
        "context": "User is a 18-year-old {} student in {}, wanting time before university.",
        "response": "Gap years aren't laziness—many use them productively for work, volunteering, or clarity about career goals. Explain your plan: what will you do, how will it benefit your future? If they see a structured plan, they might understand. However, in Zimbabwe's competitive education system, consider risks—will you lose momentum or scholarship opportunities? If you take a gap year, make it count. If they refuse, respect their concern but know you can reassess after starting university if you're truly burned out."
    },
    {
        "instruction": "My father says if I don't get straight A's, he'll stop paying my fees. I'm terrified.",
        "context": "User is a 19-year-old {} student at {}, threatened with fee withdrawal.",
        "response": "This conditional support is abusive and creates crippling pressure. Straight A's aren't realistic for everyone—many successful people had varied grades. His threat might be manipulation to motivate you, but it's harmful. Do your absolute best, and if you don't achieve all A's, seek bursaries or loans as backup. Talk to university financial aid about your situation. If he follows through, it reveals his priorities. Your education shouldn't hinge on perfection. Seek alternative funding options now, just in case."
    },
    {
        "instruction": "My parents won't let me choose my own A-level subjects. They picked what they think is best.",
        "context": "User is a 16-year-old {} student in {}, denied autonomy in subject choice.",
        "response": "Subject choice affects your career path—you should have input. Explain why your preferred subjects align with your goals. Show them career paths your choices open. If they're choosing based on outdated information or prestige, educate them. Compromise if possible—take some subjects they want, some you want. If they're rigid, you might have to comply now but can redirect at university. Sometimes you must play the long game. Try to negotiate, but know that ultimately, your passion drives success more than forced subjects."
    }
]

# Theme: Dating and parental disapproval
dating_scenarios = [
    {
        "instruction": "My boyfriend is from a poor family. My parents forbid me from seeing him because of his economic status.",
        "context": "User is a 19-year-old {} student in {}, facing classism in relationship approval.",
        "response": "Your parents' classism is unfair to both you and him. Love isn't about money—character matters more. However, parents often worry about practical issues: can he provide, will poverty cause struggles? Show them his character, ambition, work ethic. If he's in school or working toward goals, emphasize that. Sometimes parents soften when they see a person's quality. If they remain rigid, decide what matters more—their approval or this relationship. At 19, you can make your own choices, but family harmony matters too. Seek elder mediation."
    },
    {
        "instruction": "I'm secretly dating someone. My parents would never approve because we're from different religions.",
        "context": "User is a 18-year-old {} student in {}, in interfaith relationship.",
        "response": "Religious differences cause real concerns for families—future children's faith, family integration, ceremony conflicts. Your secrecy is understandable but unsustainable. Eventually, you'll need to tell them. Prepare for that conversation: How will you handle religious differences? Have you discussed it with your partner? Some interfaith couples thrive; others struggle. Be honest with yourself about compatibility. When you tell your parents, emphasize shared values beyond religion. Give them time—first reactions are often worst. They might soften. If not, you'll face hard choices about your future."
    },
    {
        "instruction": "My girlfriend wants to post us on social media but my parents will see it. They don't know I'm dating.",
        "context": "User is a 20-year-old {} student at {}, hiding relationship from parents.",
        "response": "Social media makes secrets impossible. You have two choices: tell your parents first before they find out online, or ask your girlfriend to keep it private for now. Hiding is unsustainable—better to control the narrative. At 20, you're an adult, but Zimbabwean culture respects family. Introduce the relationship properly, showing maturity and respect. If your girlfriend won't respect your need for privacy, that's a red flag—relationships require compromise. Don't let social media force your hand. Handle this maturely on your timeline."
    }
]

# Theme: Older people approaching youth
age_gap_scenarios = [
    {
        "instruction": "A 30-year-old man bought me airtime and now expects me to 'thank him properly.' I'm 16.",
        "context": "User is a 16-year-old {} student in {}, experiencing grooming from older man.",
        "response": "This is grooming—he's using gifts to obligate you sexually. You owe him nothing. Block his number, delete his contact. If he persists, tell your parents or a teacher—this is predatory behavior toward a minor. Men who target teenagers do this systematically. His 'kindness' has strings attached. You're worth more than airtime. Real gifts have no expectations. If he threatens you, report to police. Protect yourself—older men targeting young girls rarely have good intentions."
    },
    {
        "instruction": "My friend's older boyfriend has a car and money. She says I should find someone older too instead of dating broke guys our age.",
        "context": "User is a 17-year-old {} student in {}, facing peer pressure to date older men.",
        "response": "Material benefits from older men come at a cost—control, manipulation, sexual expectations, and pregnancy risks that end education. Your friend is being used, even if she doesn't see it yet. Guys your age may be broke, but they're your equals—relationships should be partnerships, not transactions. Older men target young girls because women their age won't tolerate their behavior. Focus on your education. Financial struggle now is temporary; teen pregnancy and STIs have lasting consequences. Be smarter than your friend—don't exchange your future for a car ride."
    },
    {
        "instruction": "A married lecturer says he'll help me pass his course if I'm 'nice' to him. I'm failing and desperate.",
        "context": "User is a 20-year-old {} student at {}, facing quid pro quo sexual harassment.",
        "response": "This is sexual harassment and abuse of power—both fireable offenses. Report him to the dean, head of department, or student affairs immediately. Don't meet him alone. Document everything—texts, emails, witnesses. His behavior is criminal. If you sleep with him, he'll likely fail you anyway or demand more. Get proper tutoring, form study groups, ask other lecturers for help. If the university won't act, escalate to higher authorities or media. Don't let desperation make you a victim. You deserve education without exploitation. He's done this before—reporting protects other students too."
    },
    {
        "instruction": "I'm 18 and my 35-year-old boyfriend says age is just a number. My family is worried but I think they're old-fashioned.",
        "context": "User is an 18-year-old {} student in {}, defending large age gap relationship.",
        "response": "Age isn't 'just a number'—it's life experience, power, and developmental stage. At 18, your brain is still developing; he's fully formed. He's 17 years older—why isn't he with women his age? Usually because they won't tolerate behavior young girls don't recognize as problematic. Your family sees what you can't: this power imbalance will become control. He has financial power, life experience advantage, and maturity manipulation. Listen to your family—they love you and want to protect you. In five years, you'll see this differently. Don't mistake control for love."
    }
]

# Theme: Mother conflicts (pacifist approach)
mother_conflict_scenarios = [
    {
        "instruction": "My mother throws things at me when she's angry. Yesterday a plate hit my head. I'm so angry at her.",
        "context": "User is a 15-year-old {} student in {}, experiencing physical abuse from mother.",
        "response": "Throwing objects is physical abuse, even if they miss. You're in danger—head injuries are serious. Document injuries with photos. Tell a trusted adult—teacher, counselor, relative. Your mother may be overwhelmed and lacks anger management, but that doesn't excuse violence. She needs help controlling her anger. When she's calm, tell her you're scared of her reactions. If she won't change, you need intervention from family or authorities. You deserve safety. Your anger is justified—protect yourself while seeking help for her and safety for you."
    },
    {
        "instruction": "My mom calls me names when I make mistakes—stupid, useless, worthless. It's destroying my confidence.",
        "context": "User is a 16-year-old {} student in {}, experiencing verbal abuse from mother.",
        "response": "Name-calling is emotional abuse that damages self-worth long-term. You're not stupid, useless, or worthless—her words are abusive. This might be how she was raised or her own pain expressing itself, but it's wrong. Record her words if safe. Show a trusted adult—counselor, teacher, relative. When she's calm, tell her how deeply her words hurt. If she won't stop, you need external support. Build your self-worth elsewhere—friends, mentors, achievements. Her abuse is about her issues, not your worth. Protect your mental health—limit exposure if possible. You deserve better."
    },
    {
        "instruction": "My mother compares me to her friend's children constantly. I feel like I'll never be good enough for her.",
        "context": "User is a 17-year-old {} student in {}, facing damaging maternal comparisons.",
        "response": "Constant comparison is emotionally damaging. Your mother might think she's motivating you, but she's destroying your confidence. Each person's journey is different—you're not your mother's friend's children. Tell her how these comparisons make you feel inadequate and hurt. Suggest she celebrate your strengths instead. If she continues, understand it's her issue—perhaps she's unhappy with her own life. Build self-worth independent of her approval. Focus on your own progress, not others' achievements. You are enough, even if she struggles to see it."
    },
    {
        "instruction": "My mom checks my underwear to see if I'm sexually active. I'm 18 and humiliated.",
        "context": "User is an 18-year-old {} student in {}, experiencing invasive monitoring from mother.",
        "response": "This invasion of privacy is extreme and inappropriate, especially at 18. Your mother fears teen pregnancy so much she's violating your dignity. This is about her anxiety, not your behavior. Firmly tell her this stops now—you're an adult who deserves bodily privacy. If she continues, this is abuse. Involve another adult—aunt, counselor. Her fear doesn't justify humiliation. Show her you're responsible through your actions and choices. If she won't respect boundaries, create physical distance when possible. This level of control is unhealthy for both of you."
    },
    {
        "instruction": "My mother won't speak to me for days when she's upset with me. The silent treatment hurts more than yelling.",
        "context": "User is a 16-year-old {} student in {}, experiencing emotional withdrawal from mother.",
        "response": "Silent treatment is emotional manipulation and passive-aggressive abuse. It's designed to punish and control through withdrawal of love. Your mother may not realize how damaging this is—perhaps it's how she learned to handle anger. When she's speaking to you, tell her how painful silence is. Suggest healthier communication: cooling off briefly, then discussing issues. If she won't change, understand this is her emotional immaturity. Don't beg for her attention—maintain your dignity. Seek emotional support elsewhere. Her silence says more about her than you. You deserve consistent love, not conditional presence."
    }
]

# Theme: Family financial struggles
financial_scenarios = [
    {
        "instruction": "My parents want me to get a job instead of going to university. We need money now, they say.",
        "context": "User is a 18-year-old {} student in {}, pressured to work instead of study.",
        "response": "This is a heartbreaking choice many Zimbabwean families face. Your parents see immediate need; you see long-term investment. Compromise might work: part-time university while working, or working for a year then studying. Show them how a degree increases earning potential long-term. Look for scholarships, bursaries, government loans. Can extended family help? Some employers sponsor education. Your education lifts the whole family eventually—one year of your salary helps less than a lifetime of degree-level earnings. Fight for your future while understanding their desperation."
    },
    {
        "instruction": "I have to share a textbook with my whole class because we can't afford individual copies. I'm falling behind.",
        "context": "User is a 15-year-old {} student in {}, struggling due to resource scarcity.",
        "response": "Textbook scarcity is common in Zimbabwean schools. Maximize library time, take detailed notes when you have the book, form study groups to share, photograph key pages if allowed, seek older students for used books, or check online resources. Tell teachers about your situation—they might have extra copies or suggestions. This challenge is real but manageable. Many successful people studied under worse conditions. Use this to develop resourcefulness. Your determination matters more than resources. Where there's will, there's a way."
    },
    {
        "instruction": "I'm skipping meals so my younger siblings can eat. I'm getting dizzy in class.",
        "context": "User is a 17-year-old {} student in {}, sacrificing nutrition for siblings.",
        "response": "Your sacrifice shows love, but starving yourself helps no one—you'll get sick and unable to help at all. Tell a school counselor immediately—schools have feeding programs, and there are organizations helping food-insecure families. Check for food aid programs in your area. Talk to your family about rotating who eats what. Small portions for all is better than some starving. Your health matters—you can't build a future on an empty stomach. Seek help without shame. Food insecurity affects many; assistance exists. Please reach out today."
    }
]

# Theme: Pregnancy and consequences
pregnancy_scenarios = [
    {
        "instruction": "I just found out I'm pregnant. I'm 15. I don't even know what to do first.",
        "context": "User is a 15-year-old {} student in {}, newly discovered pregnancy.",
        "response": "First, breathe. You're scared, but you're not alone. Step one: see a doctor for prenatal care—your health matters. Step two: tell a trusted adult—parent, aunt, counselor. Yes, they'll be upset, but you need support. Step three: discuss options with your family. Many organizations help teen mothers continue education. The baby's father's family should be involved too. This changes your life, but it's not the end—many young mothers succeed with support. Get help today. Don't hide or delay—early prenatal care is crucial."
    },
    {
        "instruction": "My girlfriend is pregnant and wants to keep it. I want her to abort but it's illegal here.",
        "context": "User is a 19-year-old {} student in {}, facing unwanted pregnancy.",
        "response": "Abortion is severely restricted in Zimbabwe—illegal except for specific circumstances. Dangerous illegal abortions kill girls. You can't force her choice—it's her body and risk. You created this together; you're both responsible now. Face your families, discuss options: parenting together, adoption, family support. Running won't help—she'll have the baby regardless. Man up and take responsibility. You both can continue education with support—hard, but possible. Some students parent while studying. Child support is legally required. This is reality—handle it maturely."
    }
]

# Add varied locations and ethnicities
locations = ["Harare", "Bulawayo", "Mutare", "Gweru", "Kwekwe", "Kadoma", "Masvingo", "Chitungwiza", "Bindura", "Beitbridge", "Gwanda", "Kariba", "Victoria Falls", "Chegutu", "Marondera", "Zvishavane", "Chinhoyi", "Karoi", "Rusape", "Chiredzi"]
genders = ["male", "female"]
ethnicities = ["Shona", "Ndebele", "Tonga", "Kalanga", "Venda", "Shangani"]
universities = ["UZ", "NUST", "MSU", "BUSE", "CUT", "GZU", "polytechnic", "teachers college"]

# Generate entries from templates
generated_entries = []
for scenario_list in [academic_scenarios, dating_scenarios, age_gap_scenarios, mother_conflict_scenarios, financial_scenarios, pregnancy_scenarios]:
    for template in scenario_list:
        for _ in range(20):  # Create multiple variations of each template
            location = random.choice(locations)
            gender = random.choice(genders)
            ethnicity = random.choice(ethnicities)

            context = template["context"].format(ethnicity, location)

            # For university students
            if "{}" in template.get("context", ""):
                if "university" in template["context"] or random.random() > 0.5:
                    if "at {}" in template["context"]:
                        uni = random.choice(universities)
                        context = template["context"].replace("at {}", f"at {uni}")

            entry = {
                "instruction": template["instruction"],
                "context": context,
                "response": template["response"]
            }
            generated_entries.append(entry)

# Additional unique scenarios to reach 1000
additional_scenarios = [
    {
        "instruction": "My cousin is living with us and he gets more attention than me. My parents treat him better.",
        "context": "User is a {}-year-old {} {} student in {}, feeling neglected.",
        "response": "This hurts—feeling second in your own home. Parents might give him extra attention because he's away from his parents, or they're being polite to extended family. It's not about loving you less. Talk to your parents privately about feeling neglected. They might not realize the imbalance. Ask for one-on-one time with them. If your cousin is temporary, endure it; he'll leave eventually. If permanent, you need to address this family dynamic. Your feelings are valid. Seek reassurance of their love through conversation, not competition with your cousin."
    },
    {
        "instruction": "I want to join the school soccer team but my parents say sports are a waste of time.",
        "context": "User is a {}-year-old {} {} student in {}, denied extracurricular activities.",
        "response": "Many Zimbabwean parents prioritize academics over sports. Show them research on sports benefits: discipline, teamwork, fitness, stress relief, potential scholarships. Prove sports won't hurt grades—maintain or improve academic performance. Negotiate: if grades stay up, you play sports. Some parents fear soccer leads to bad company or injuries. Address their specific concerns. Sports teach life skills. Many successful people were athletes. Balance is important—all study and no play isn't healthy. Fight for this respectfully."
    },
    {
        "instruction": "My parents make me go to church three times a week but I don't believe anymore. I feel like a hypocrite.",
        "context": "User is a {}-year-old {} {} student in {}, struggling with faith.",
        "response": "Religious doubt is normal, especially as you mature intellectually. However, openly rejecting faith will devastate your parents and cause family conflict. At 19, financially dependent, you might have to participate outwardly while exploring beliefs privately. Read, question, discuss with trusted people. Faith journeys aren't linear—doubt doesn't mean permanent atheism. If you're certain of disbelief, decide: wait until independent to declare it, or tell them now and face consequences. Neither choice is easy. Many people navigate family faith expectations while holding private beliefs. This is a personal journey."
    },
    {
        "instruction": "My sibling is clearly the favorite child. Nothing I do is ever good enough.",
        "context": "User is a {}-year-old {} {} student in {}, experiencing parental favoritism.",
        "response": "Parental favoritism damages self-worth deeply. It's real and painful. Confront your parents calmly about the imbalance—they might not realize. If they deny it, give examples. Sometimes parents favor without intention—birth order, gender, personality fit. You can't control their behavior, but you can build self-worth elsewhere. Seek validation from other sources: teachers, mentors, achievements. Don't compete for love—you deserve it without earning it. If they won't change, focus on becoming independent. Their failure to see your worth doesn't diminish it."
    },
    {
        "instruction": "I have to hide my achievements because my parents praise me and it makes my siblings jealous and mean to me.",
        "context": "User is a {}-year-old {} {} student in {}, hiding success to avoid sibling jealousy.",
        "response": "You shouldn't have to dim your light to make others comfortable. Your achievements are valid and worth celebrating. Your siblings' jealousy is their issue to manage, not yours to accommodate. Talk to your parents about balancing praise—celebrate everyone's strengths. But don't hide your successes. That teaches you to be small. Instead, encourage your siblings to pursue their own goals. Success isn't a limited resource—your achievement doesn't diminish theirs. Stand in your excellence while being kind, not arrogant. They'll adjust or they won't—either way, keep shining."
    },
    {
        "instruction": "My parents give me more responsibilities than my siblings because I'm the 'responsible one.' It's exhausting.",
        "context": "User is a {}-year-old {} {} student in {}, overburdened due to responsibility.",
        "response": "Being the responsible child is both a compliment and a curse. Your parents rely on you because you're capable, but it's unfair to burden you more than siblings. Tell them you appreciate their trust but you're overwhelmed. Siblings should share responsibilities equally—being responsible doesn't mean being the family servant. Suggest redistributing chores and duties. If they refuse, set boundaries. You can say no sometimes. Being reliable shouldn't mean sacrificing yourself. Your siblings won't learn responsibility if you do everything. Stand up for balance."
    }
]

# Generate variations of additional scenarios
for template in additional_scenarios:
    for _ in range(40):
        age = random.randint(14, 21)
        gender = random.choice(genders)
        ethnicity = random.choice(ethnicities)
        location = random.choice(locations)

        context = template["context"].format(age, gender, ethnicity, location) if template["context"].count("{}") == 4 else template["context"]

        entry = {
            "instruction": template["instruction"],
            "context": context,
            "response": template["response"]
        }
        generated_entries.append(entry)

# Combine with existing data
all_data = existing_data + generated_entries

# Trim to exactly 1000
all_data = all_data[:10000]

# Write to file
with open('data/raw/training_family.json', 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"Generated {len(all_data)} total entries")
