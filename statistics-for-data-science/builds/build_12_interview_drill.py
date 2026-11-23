# %% [markdown]
# # Build 12: Interview drill (45 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# No dataset this time. Ten timed questions, three minutes each, then one case question in fifteen minutes.
# Do not look anything up during the timed part: the point is to find out what you can explain from memory.
# For each answer: a short definition, a concrete example (ideally from one of your PetalMart builds) and one
# common mistake. Write each answer in the text cell under its question.
#
# When all eleven are done, and not before, open the model answers in the Academy and mark each answer:
# right, missed something, or could be clearer.

# %%
import datetime as dt

QUESTIONS = [
    "What does a p-value of 0.03 mean, and what does it not mean?",
    "When would you report the median instead of the mean? Use PetalMart's order values.",
    "A colleague says: 'The 95% confidence interval for the lift is 0.5 to 3.1 points, so there is a 95% chance the "
    "true lift is in there.' Is that right? Explain.",
    "What is statistical power, and what would you change to get more of it?",
    "You test twenty segments and one comes out significant. What do you do?",
    "Explain the Central Limit Theorem to a product manager, and why it lets you use a t-test on skewed revenue.",
    "In a regression of revenue on marketing spend, the spend coefficient falls from 3.2 to 2.1 when you add season. Why?",
    "Interpret an odds ratio of 1.8 for complaints in a churn model. What is wrong with 'complaints raise churn by 80%'?",
    "When would you choose a Bayesian analysis over a frequentist one for a business decision?",
    "Why can't you just compare cities that got same-day delivery with cities that did not, and what would you do instead?",
]
CASE = ("PetalMart's checkout conversion fell from 65% to 61% last week. Write, step by step, how you would investigate: "
        "the data and the metric definition first, then segments (device, app version, region, payment method, new versus "
        "returning), recent releases and experiments, outside causes (festival, outage, payment partner), and how you would "
        "tell a real change from noise. Finish with what you would tell the product head today, and what you would do next.")


def ask(i: int, minutes: int = 3) -> None:
    """Show question i (1 to 10, or 11 for the case) and the time to stop writing."""
    text, minutes = (CASE, 15) if i == 11 else (QUESTIONS[i - 1], minutes)
    stop = (dt.datetime.now() + dt.timedelta(minutes=minutes)).strftime("%H:%M:%S")
    print(f"Question {i} of 11 · stop writing at {stop}\n\n{text}")


# %%
ask(1)

# %% [markdown]
# Your answer to question 1:

# %%
ask(2)

# %% [markdown]
# Your answer to question 2:

# %%
ask(3)

# %% [markdown]
# Your answer to question 3:

# %%
ask(4)

# %% [markdown]
# Your answer to question 4:

# %%
ask(5)

# %% [markdown]
# Your answer to question 5:

# %%
ask(6)

# %% [markdown]
# Your answer to question 6:

# %%
ask(7)

# %% [markdown]
# Your answer to question 7:

# %%
ask(8)

# %% [markdown]
# Your answer to question 8:

# %%
ask(9)

# %% [markdown]
# Your answer to question 9:

# %%
ask(10)

# %% [markdown]
# Your answer to question 10:

# %%
ask(11)

# %% [markdown]
# Your answer to the case question:

# %% [markdown]
# ## After the drill
# List the three topics you most need to revise and go back to those lessons this week. Try the drill again in a few
# days and see how much faster and clearer you have become.
