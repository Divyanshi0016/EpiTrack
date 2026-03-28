"""
engine.py
---------
Chatbot engine.
- OpenAIEngine  : uses GPT-4o-mini (needs API key)
- FallbackEngine: fully offline, rule-based, covers all demo queries
"""

import os
import pandas as pd
from src.chatbot.context_builder import (
    load_summary, build_context,
    get_country_detail, get_safe_countries, get_high_risk_countries
)

SYSTEM_PROMPT = """
You are EpiBot, an expert epidemic risk assistant for EpiTrack.
You have real-time epidemic data: risk scores, R0 estimates, case trends,
and vaccination rates for countries worldwide.

Rules:
- Be factual. Only use provided context data.
- Keep answers under 200 words unless a report is requested.
- Use bullet points for lists.
- Never make up statistics.
- Tone: calm, professional, like a public health expert.
"""


class OpenAIEngine:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.model   = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _client_(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def ask(self, user_query: str,
            df: pd.DataFrame = None,
            chat_history: list = None) -> str:
        context  = build_context(df)
        messages = [{"role": "system",
                     "content": SYSTEM_PROMPT + f"\n\nDATA:\n{context}"}]

        if chat_history:
            for role, msg in chat_history[-6:]:
                messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": msg
                })

        messages.append({"role": "user", "content": user_query})

        try:
            resp = self._client_().chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=500, temperature=0.4
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return (f"OpenAI error: {e}\n\n"
                    + FallbackEngine().ask(user_query, df))


class FallbackEngine:
    """Fully offline rule-based engine. No API key needed."""

    def ask(self, user_query: str,
            df: pd.DataFrame = None,
            chat_history: list = None) -> str:

        if df is None:
            try:
                df = load_summary()
            except FileNotFoundError:
                return (
                    "⚠️ Risk summary data not found.\n\n"
                    "**Fix:** Run this in your terminal:\n"
                    "```\npython -m src.chatbot.risk_summary\n```\n"
                    "Or click **Refresh Risk Data** in the sidebar."
                )

        q = user_query.lower()

        # Safe / travel
        if any(w in q for w in ['safe', 'safest', 'travel', 'visit', 'where can', 'go to']):
            safe = get_safe_countries(df, 8)
            high = get_high_risk_countries(df, 5)
            return (
                "**✅ Safe destinations (Low Risk):**\n"
                + "\n".join(f"  🟢 {c}" for c in safe)
                + "\n\n**⚠️ Avoid these High Risk areas:**\n"
                + "\n".join(f"  🔴 {c}" for c in high)
                + "\n\n*Based on R0, 7-day growth rate, and vaccination data.*"
            )

        # High risk
        if any(w in q for w in ['high risk', 'danger', 'worst', 'avoid', 'hotspot']):
            high = df[df['status'] == 'High Risk'].head(10)
            if high.empty:
                return "✅ No countries currently classified as High Risk."
            lines = [f"  🔴 **{r['country']}** — R0={r['r0_estimate']:.2f}, {r['trend']}"
                     for _, r in high.iterrows()]
            return "**🔴 High Risk Countries:**\n" + "\n".join(lines)

        # Rising
        if any(w in q for w in ['rising', 'increasing', 'getting worse', 'deteriorat']):
            rising = df[df['trend'] == 'Rising'].head(10)
            if rising.empty:
                return "✅ No countries currently show a rising trend."
            lines = [f"  📈 **{r['country']}** — growth={r['growth_rate']*100:+.1f}%, R0={r['r0_estimate']:.2f}"
                     for _, r in rising.iterrows()]
            return "**📈 Countries with Rising Trends:**\n" + "\n".join(lines)

        # Declining
        if any(w in q for w in ['declining', 'improving', 'better', 'decreas']):
            dec = df[df['trend'] == 'Declining'].head(10)
            if dec.empty:
                return "No countries currently show a declining trend."
            lines = [f"  📉 **{r['country']}** — growth={r['growth_rate']*100:+.1f}%, R0={r['r0_estimate']:.2f}"
                     for _, r in dec.iterrows()]
            return "**📉 Countries Improving (Declining Trend):**\n" + "\n".join(lines)

        # Summary / report
        if any(w in q for w in ['summary', 'overview', 'situation', 'report', 'global', 'world']):
            n_high = len(df[df['status'] == 'High Risk'])
            n_mod  = len(df[df['status'] == 'Moderate Risk'])
            n_low  = len(df[df['status'] == 'Low Risk'])
            n_rise = len(df[df['trend']  == 'Rising'])
            n_dec  = len(df[df['trend']  == 'Declining'])
            top3h  = df[df['status'] == 'High Risk'].head(3)['country'].tolist()
            top3s  = df[df['status'] == 'Low Risk'].head(3)['country'].tolist()
            return (
                f"**🌍 Global Epidemic Summary**\n\n"
                f"📊 Countries monitored: **{len(df)}**\n"
                f"🔴 High Risk  : **{n_high}**\n"
                f"🟡 Moderate   : **{n_mod}**\n"
                f"🟢 Low Risk   : **{n_low}**\n\n"
                f"📈 Rising trends  : **{n_rise}** countries\n"
                f"📉 Declining trends: **{n_dec}** countries\n\n"
                f"⚠️ Most critical : {', '.join(top3h) if top3h else 'None'}\n"
                f"✅ Safest right now: {', '.join(top3s) if top3s else 'None'}"
            )

        # R0 explanation
        if any(w in q for w in ['r0', 'r-zero', 'reproduction', 'what is r']):
            avg = df['r0_estimate'].mean()
            return (
                "**What is R0?**\n\n"
                "R0 = how many people one infected person spreads the disease to.\n\n"
                "- **R0 > 1** → outbreak **growing**\n"
                "- **R0 = 1** → outbreak **stable**\n"
                "- **R0 < 1** → outbreak **declining**\n\n"
                f"**Current global average R0: {avg:.2f}**\n"
                f"{'⚠️ Above 1 — epidemic spreading globally.' if avg > 1 else '✅ Below 1 — situation improving.'}"
            )

        # Vaccination
        if any(w in q for w in ['vaccin', 'jab', 'booster', 'immunis']):
            top_v = df.nlargest(5, 'vax_rate')[['country', 'vax_rate']]
            low_v = df.nsmallest(5, 'vax_rate')[['country', 'vax_rate']]
            return (
                "**💉 Vaccination Coverage:**\n\n"
                "**Highest:**\n"
                + "\n".join(f"  💉 {r['country']}: {r['vax_rate']:.1f}/100"
                            for _, r in top_v.iterrows())
                + "\n\n**Lowest:**\n"
                + "\n".join(f"  ⚠️ {r['country']}: {r['vax_rate']:.1f}/100"
                            for _, r in low_v.iterrows())
            )

        # Specific country lookup
        for country in df['country'].values:
            if country.lower() in q:
                detail = get_country_detail(country, df)
                r      = df[df['country'] == country].iloc[0]
                advice = {
                    'High Risk':     "\n⚠️ **Travel Advice:** Avoid non-essential travel.",
                    'Moderate Risk': "\n🟡 **Travel Advice:** Exercise caution. Monitor situation.",
                    'Low Risk':      "\n✅ **Travel Advice:** Currently considered safe.",
                }.get(r['status'], "")
                return f"**{country} — Epidemic Status**\n\n```\n{detail}```{advice}"

        # Default
        safe = get_safe_countries(df, 5)
        return (
            "I can help with epidemic questions. Try:\n\n"
            "- *'Which countries are safe to travel?'*\n"
            "- *'Show high risk countries'*\n"
            "- *'What is the situation in India?'*\n"
            "- *'Give me a global summary'*\n"
            "- *'Which countries are improving?'*\n"
            "- *'Explain R0'*\n\n"
            f"Currently safe: {', '.join(safe)}"
        )


def generate_report(df: pd.DataFrame = None, engine=None) -> str:
    if df is None:
        df = load_summary()

    if engine and isinstance(engine, OpenAIEngine):
        return engine.ask(
            "Generate a full structured epidemic report with: executive summary, "
            "top 5 high-risk countries, top 5 safe countries, key trends, "
            "and public health recommendations.",
            df=df
        )

    high = df[df['status'] == 'High Risk'].head(5)
    safe = df[df['status'] == 'Low Risk'].head(5)
    rise = df[df['trend']  == 'Rising'].head(5)
    dec  = df[df['trend']  == 'Declining'].head(5)

    def tbl(sub, cols=('country','risk_score','r0_estimate','trend')):
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(['---']*len(cols)) + " |"
        rows   = "\n".join(
            "| " + " | ".join(str(r[c]) for c in cols) + " |"
            for _, r in sub.iterrows()
        )
        return f"{header}\n{sep}\n{rows}"

    return f"""# 🦠 EpiTrack Epidemic Report
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Countries monitored:** {len(df)}

---

## Executive Summary
- **{len(df[df['status']=='High Risk'])}** High Risk countries
- **{len(df[df['trend']=='Rising'])}** countries with rising trends
- **{len(df[df['status']=='Low Risk'])}** Low Risk (safe) countries

---

## ⚠️ Top High Risk Countries
{tbl(high)}

---

## ✅ Safest Countries
{tbl(safe)}

---

## 📈 Rising Trends
{"".join(f"- **{r['country']}**: {r['growth_rate']*100:+.1f}%, R0={r['r0_estimate']:.2f}\n" for _,r in rise.iterrows())}

---

## 📉 Declining Trends (Improving)
{"".join(f"- **{r['country']}**: {r['growth_rate']*100:+.1f}%, R0={r['r0_estimate']:.2f}\n" for _,r in dec.iterrows())}

---

## 💡 Recommendations
1. Avoid non-essential travel to High Risk countries
2. Exercise caution in Moderate Risk countries
3. Ensure vaccinations are up to date before travel
4. Follow local health guidelines at all times
"""


def get_engine(api_key: str = None):
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if key and key not in ("", "YOUR_API_KEY", "sk-your-key-here"):
        return OpenAIEngine(api_key=key)
    return FallbackEngine()
