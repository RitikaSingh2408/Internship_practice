import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# ====== Data Shape ====

print(matches.shape)
print(deliveries.shape)

# ====== First 5 Rows =======

# print(matches.head())
# print(deliveries.head())

# ====== Dataset info =======

# print(matches.info())
# print(deliveries.info())

# ==== Missing Values =====

# print(matches.isnull().sum())
# print(deliveries.isnull().sum())

# ====== Duplicate Values ======

# print(matches.duplicated().sum())

# print(deliveries.duplicated().sum())

# ======= Statistical Summary =======

# print(matches.describe())
# print(deliveries.describe())

# ===== fill missing values =====

# print(matches['winner'].fillna("No Result", inplace=True))

# print(matches['player_of_match'].fillna("Unknown", inplace=True))

# print(matches['city'].fillna("Unknown", inplace=True))

# ===== remove duplicate =====

# print(matches.drop_duplicates(inplace=True))

# print(deliveries.drop_duplicates(inplace=True))

# ===== kpi1 Total matches =====

total_matches = matches.shape[0]
print("Total Matches:", total_matches)

# ==== visualization ====

plt.figure(figsize=(4,4))
plt.bar(["matches"],[total_matches])
plt.title("Total IPL Matches")
plt.ylabel("Count")
plt.show()

# ==== kpi2 Highest Run Scorer ====

runs = deliveries.groupby('batsman')['batsman_runs'].sum()

highest_run = runs.sort_values(ascending=False)

print(highest_run.head(10))

# ==== visualization ====

highest_run.head(10).plot(
    kind='bar',
    figsize=(10,5)
)
plt.title("Top 10 run scorers")
plt.xlabel("Player")
plt.ylabel("Runs")
plt.show()

# ==== they check a list of columns in the dataset ====
# print(deliveries.columns.tolist())

# ==== kpi3 Highest Wicket Taker ====
wickets = deliveries[deliveries['player_dismissed'].notna()]

highest_wicket = wickets.groupby('bowler').size().sort_values(ascending=False)
print(highest_wicket.head(10))

# ==== visualization ====
highest_wicket.head(10).plot(
    kind='bar',
    figsize=(10,5),
    color='orange'
)
plt.title("Top 10 Wicket Takers")
plt.xlabel("Bowler")
plt.ylabel("Wickets")
plt.xticks(rotation=45)
plt.show()

# ===== kpi4 team wins =====
team_wins = matches['winner'].value_counts()
print(team_wins)

# ==== visualization ====
team_wins.plot(
    kind ='bar',
    figsize=(12,6),
    color='green'
)

plt.title("Teams Wins")
plt.ylabel("wins")
plt.xlabel("Teams")
plt.show()

# === kpi5 Venue Analysis ====
venue = matches['venue'].value_counts()
print(venue.head(10))

# ==== visualization ====
venue.head(10).plot(
    kind='bar',
    figsize=(12,5),
    color='purple'
)
plt.title("Top venues by matches")
plt.ylabel("Matches")
plt.xticks(rotation=45)
plt.show()

# === kpi6 Toss Analysis ====
toss = matches.groupby('toss_winner')['winner'].count()
print(toss)

# === toss winner wins match===
matches['toss_match_win'] = np.where(
    matches['toss_winner'] ==matches['winner'],
    "yes",
    "No"
)
print(matches['toss_match_win'].value_counts())

# ==== visualization ====
matches['toss_match_win'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6),
    color='blue'
)
plt.title("Toss Winner Wins Match")
plt.ylabel("")
plt.show()

# ===== orange cap analysis ====
orange_cap = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False)
print(orange_cap.head())

# # ==== visualization ====
orange_cap.head(10).plot(kind='bar',figsize=(8,5), color='red')
plt.title("Orange cap Holders")
plt.ylabel("Runs")
plt.show()

# ===== purple cap =====

purple_cap = wickets.groupby('bowler').size().sort_values(ascending=False)

print(purple_cap.head())

# # ==== visualization ====
purple_cap.head(5).plot(kind='bar',figsize=(8,5), color='purple')
plt.title("Purple cap Holders")
plt.ylabel("Wickets")
plt.show()

# ===== team wise winning percentage =====

matches_played = pd.concat([
    matches['team1'],
    matches['team2']
]).value_counts()

matches_won = matches['winner'].value_counts()

winning_percentage = (matches_won / matches_played * 100).round(2)

print(winning_percentage.sort_values(ascending=False))

winning_percentage.sort_values(ascending=False).plot(
    kind='bar',
    figsize=(12,5)
)

plt.title("Team Wise Winning Percentage")
plt.ylabel("Winning %")
plt.show()

# ===== toss winner vs match winner ======

matches['Toss Match Win'] = np.where(
    matches['toss_winner'] == matches['winner'],
    'yes',
    'No'
)
print(matches['Toss Match Win'].value_counts())

# # === visualization ====
matches['Toss Match Win'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6),
    color='blue'
)
plt.title("Toss Winner Wins Match")
plt.ylabel("")
plt.show()

# ===== season wise champion =====

champions = matches.groupby('season').last()['winner']

print(champions)

# # === visualization ====
champions.value_counts().plot(
    kind='bar',
    figsize=(8,5),
    color='green'
)
plt.title("Season Wise Champions")
plt.ylabel("Champions")
plt.show()

# == player of the match analysis ====

pom =matches['player_of_match'].value_counts()
print(pom.head(10))

pom.head(10).plot(
    kind='bar',
    figsize=(10,5),
    color='brown'
)

plt.title("Top Player Of the Match winners")
plt.ylabel("Awards")
plt.show()