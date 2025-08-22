"""
Simple Telegram Bot for Benjamin AI System
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters,
    ContextTypes,
    CallbackQueryHandler,
)
from telegram.constants import ParseMode, ChatAction
import requests

from ..config.settings import settings


class BenjaminTelegramBot:
    """Simple Telegram Bot for Benjamin AI System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.application: Optional[Application] = None
        self.data_api_url = settings.data_api_base_url
        # Track last message timestamps per chat for session end
        self._last_msg_ts: Dict[str, float] = {}
        # Ephemeral session ids per chat to group short-term conversation turns
        self._session_ids: Dict[str, str] = {}
        # Per-chat debug toggle to include context_debug information from Agentic API
        self._debug_mode: Dict[str, bool] = {}
        # Track workout change request state per chat
        self._awaiting_workout_change: Dict[str, bool] = {}
        
        # Main menu keyboard (streamlined)
        self.main_menu_keyboard = ReplyKeyboardMarkup([
            [KeyboardButton("📊 Today's Data"), KeyboardButton("🏃 Recent Activities")],
            [KeyboardButton("🤖 Training of the Day"), KeyboardButton("🧹 End Session")],
            [KeyboardButton("🔄 Sync Data")], [KeyboardButton("🔧 Status")],
            [KeyboardButton("❓ Help")]
        ], resize_keyboard=True)
        
    def setup_application(self) -> Application:
        """Setup the Telegram application"""
        if not settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        # Create application
        self.application = Application.builder().token(settings.telegram_bot_token).build()
        
        # Register handlers
        self._register_handlers()
        # Configure persistent command menu in the background
        try:
            asyncio.get_event_loop().create_task(self._configure_bot_menu())
        except Exception:
            pass
        
        return self.application
    
    def _register_handlers(self):
        """Register all command and message handlers"""
        app = self.application
        
        # Command handlers
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("data", self.data_command))
        app.add_handler(CommandHandler("activities", self.activities_command))
        app.add_handler(CommandHandler("sync", self.sync_command))
        app.add_handler(CommandHandler("discussion", self.discussion_command))
        app.add_handler(CommandHandler("end", self.end_command))
        app.add_handler(CommandHandler("debug", self.debug_command))
        
        # Callback buttons
        app.add_handler(CallbackQueryHandler(self._on_button_pressed))
        
        # Message handlers
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
        # Unknown commands fallback (e.g., /end-session)
        app.add_handler(MessageHandler(filters.COMMAND, self._unknown_command))
        
        # Error handler
        app.add_error_handler(self.error_handler)
        # Background session watchdog removed

    async def end_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /end and /end_session commands: flush session and acknowledge to user."""
        chat_id = str(update.effective_chat.id) if update and update.effective_chat else None
        # Best-effort notify Agentic API to summarize and clear session memory
        if chat_id:
            try:
                sid = self._session_ids.get(chat_id)
                requests.post(
                    f"{settings.agentic_api_base_url}/end-session",
                    json={"reason": "user_request", "session_id": sid},
                    timeout=10,
                )
            except Exception:
                pass
            # Clear local inactivity tracker
            try:
                if chat_id in self._last_msg_ts:
                    del self._last_msg_ts[chat_id]
                if chat_id in self._session_ids:
                    del self._session_ids[chat_id]
            except Exception:
                pass
        try:
            await update.message.reply_text(
                "🧹 Session ended. You can start a new one any time with /start or by sending a message.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        welcome_message = f"""🎉 *Welcome to Benjamin AI Coach!*

Hello {user.first_name}! I'm your personal AI coaching assistant, here to help you achieve your fitness goals with data-driven insights and personalized recommendations.

🤖 *What I can do for you:*
• 📊 Daily health & activity summaries
• 🏃 Personalized workout recommendations  
• 📈 Trend analysis and insights

🚀 *Getting Started:*
Use the menu below or try these commands:
• `/data` - See today's health metrics
• `/activities` - View recent workouts
• `/help` - Full command list

Ready to optimize your training? Let's go! 💪"""
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.main_menu_keyboard
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command with comprehensive, self-documenting menu"""
        help_text = (
            "❓ *Benjamin AI Coach — Help & Commands*\n\n"
            "*Essentials:*\n"
            "• `/start` — Welcome and main menu\n"
            "• `/help` — Show this help\n"
            "• `/discussion` — Start AI team discussion (Workout of the Day)\n"
            "• `/end` — End current session\n\n"
            "*Data & Status:*\n"
            "• `/data` — Today's health metrics\n"
            "• `/activities` — Recent workouts\n"
            "• `/status` — System status\n"
            "• `/sync` — Trigger data sync\n\n"
            ""
            "You can also just chat naturally. Try: 'What's my VO2max?' or 'I feel tired; can we do an easy gravel ride?'"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_menu_keyboard)
    
    async def data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /data command - show today's health data (dashboard-aware)."""
        try:
            # Prefer agent dashboard for richer context
            dash = None
            try:
                resp_dash = requests.get(f"{self.data_api_url}/dashboard/agent-context", timeout=settings.data_api_timeout)
                if resp_dash.status_code == 200:
                    dash = resp_dash.json()
            except Exception:
                dash = None

            if isinstance(dash, dict) and dash.get("latest_health"):
                message = self._format_today_from_dashboard(dash)
            else:
                # Fallback: latest-only
                response = requests.get(
                    f"{self.data_api_url}/health-data/latest",
                    timeout=settings.data_api_timeout
                )
                if response.status_code == 200:
                    health_data = response.json()
                    message = self._format_health_data(health_data)
                else:
                    message = "❌ Unable to fetch health data. Please try again later."

        except Exception as e:
            self.logger.error(f"Error fetching health data: {e}")
            message = "⚠️ Error accessing health data. Please check the system status."

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    
    async def activities_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /activities command - show recent activities"""
        try:
            # Fetch recent activities from Data API
            response = requests.get(
                f"{self.data_api_url}/activities/recent?days=7&limit=10",
                timeout=settings.data_api_timeout
            )
            
            if response.status_code == 200:
                activities = response.json()
                message = self._format_activities(activities)
            else:
                message = "❌ Unable to fetch activities. Please try again later."
        
        except Exception as e:
            self.logger.error(f"Error fetching activities: {e}")
            message = "⚠️ Error accessing activity data. Please check the system status."
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show system status"""
        try:
            # Check Data API health
            response = requests.get(
                f"{self.data_api_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                status_data = response.json()
                message = f"""🟢 *System Status: HEALTHY*

📊 *Data API:* ✅ Running
🕐 *Last Updated:* {status_data.get('timestamp', 'Unknown')}
🗄️ *Database:* {status_data.get('database', 'Unknown')}

All systems are operational! 🚀"""
            else:
                message = "🟡 *System Status: DEGRADED*\n\nData API is experiencing issues."
        
        except Exception as e:
            self.logger.error(f"Error checking system status: {e}")
            message = "🔴 *System Status: ERROR*\n\nUnable to connect to backend services."
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    def _to_html_safe(self, text: str) -> str:
        """Convert simple Markdown-like fences to Telegram HTML-safe text."""
        import re
        if not text:
            return ""
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        safe = re.sub(r"```(?:json)?\n([\s\S]*?)```", lambda m: f"<pre>{m.group(1)}</pre>", safe)
        return safe

    async def discussion_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /discussion command. Show intent menu to guide routing, with option to run team discussion."""
        try:
            await update.message.reply_text(
                "🤖 What do you need today?",
                reply_markup=self._build_coach_intents_keyboard()
            )
            return
        except Exception:
            pass
        # Fallback to direct discussion if inline menus fail
        await update.message.reply_text("🤖 Starting AI team discussion... this may take ~10s")
        try:
            resp = requests.post(
                f"{settings.agentic_api_base_url}/daily-discussion",
                json={},
                timeout=1800
            )
        except requests.RequestException as e:
            self.logger.error(f"Agentic API network error: {e}")
            await update.message.reply_text("⚠️ Could not contact Agentic API (network).")
            return

        if resp.status_code != 200:
            self.logger.error(f"Agentic API returned {resp.status_code}: {resp.text[:200]}")
            await update.message.reply_text("❌ Agentic API error. Try again later.")
            return

        try:
            data = resp.json()
        except Exception as e:
            self.logger.error(f"Agentic API JSON parse error: {e}; body head: {resp.text[:120]}")
            await update.message.reply_text("⚠️ Discussion failed. Please try again in a moment.")
            return
        preview = data.get("telegram_message", "")
        if len(preview) > 3500:
            preview = preview[:3500] + "..."
        # Seeding assistant message removed
        # Send as Telegram Markdown without HTML escaping
        try:
            await update.message.reply_text(
                f"✅ Discussion complete!\n\n{preview}",
                parse_mode=ParseMode.MARKDOWN
                )
        except Exception as e:
            self.logger.error(f"Telegram Markdown formatting error (fallback to plain): {e}")
            plain = preview.replace("```", "").replace("*", "").replace("_", "").replace("`", "")
            await update.message.reply_text(
                f"✅ Discussion complete! (plain text)\n\n{plain}",
                parse_mode=None
            )
    
    async def sync_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sync command - trigger data synchronization"""
        await update.message.reply_text("🔄 Starting data synchronization...")
        
        try:
            # Trigger sync via Data API
            response = requests.post(
                f"{self.data_api_url}/sync/trigger",
                params={"days_back": 3},
                timeout=60
            )
            
            if response.status_code == 200:
                sync_result = response.json()
                message = f"""✅ *Sync Completed Successfully!*

📊 *Results:*
• Garmin Health: {sync_result.get('results', {}).get('garmin_health', {}).get('records', 0)} records
• Strava Activities: {sync_result.get('results', {}).get('strava_activities', {}).get('records', 0)} records

🕐 *Completed:* {datetime.now().strftime('%H:%M')}"""
            else:
                message = "❌ Sync failed. Please try again later."
        
        except Exception as e:
            self.logger.error(f"Error triggering sync: {e}")
            message = "⚠️ Error triggering sync. Please check the system status."
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
      
    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        message_text = update.message.text
        # Update last activity timestamp
        if update and update.effective_chat:
            self._last_msg_ts[str(update.effective_chat.id)] = datetime.now().timestamp()
        
        # Handle quick reply buttons
        if message_text == "📊 Today's Data":
            await self.data_command(update, context)
        elif message_text == "🏃 Recent Activities":
            await self.activities_command(update, context)
        elif message_text == "🤖 Training of the Day":
            await self.discussion_command(update, context)
        elif message_text == "🧹 End Session":
            await self.end_command(update, context)
        elif message_text == "🔄 Sync Data":
            await self.sync_command(update, context)
        elif message_text == "🔧 Status":
            await self.status_command(update, context)
        elif message_text == "❓ Help":
            await self.help_command(update, context)
        else:
            # Intercept goal/event flows
            chat_id = str(update.effective_chat.id) if update and update.effective_chat else None
            # Ensure a session id for this chat
            session_id = self._ensure_session_id(chat_id) if chat_id else None
            # Show typing indicator while we process (non-blocking)
            try:
                if chat_id:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            # Coach intent: request different workout
            if chat_id and self._awaiting_workout_change.get(chat_id):
                self._awaiting_workout_change.pop(chat_id, None)
                # Route user's preference to PM
                try:
                    resp = requests.post(
                        f"{settings.agentic_api_base_url}/route",
                        json={"text": message_text, "session_id": session_id},
                        timeout=600,
                    )
                    if resp.status_code == 200:
                        data = resp.json() or {}
                        ans = data.get("reply") or ""
                        if ans:
                            try:
                                await update.message.reply_text(ans, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🔄 Request Different", callback_data="request_different")],
                                    [InlineKeyboardButton("💬 Chat with Coach", callback_data="chat_coach")],
                                ]))
                            except Exception:
                                await update.message.reply_text(ans)
                            return
                except Exception:
                    pass
                await update.message.reply_text("⚠️ Couldn't reach the coach. Try again in a moment.")
                return
            # Goals/events flows removed
            # Handle general conversation
            await self._handle_general_message(update, context)
    
    def _format_health_data(self, health_data: Dict[str, Any]) -> str:
        """Format health data into a readable message"""
        if not health_data:
            return "📊 No health data available for today."
        
        date = health_data.get('date', 'Unknown')
        
        message = f"📊 *Health Data for {date}*\n\n"
        
        # Sleep metrics
        if health_data.get('sleep_score'):
            message += f"😴 *Sleep Score:* {health_data['sleep_score']}/100\n"
        if health_data.get('sleep_hours'):
            message += f"🕐 *Sleep Duration:* {self._format_hours_as_hhmm(health_data['sleep_hours'])}\n"
        
        # Recovery metrics
        if health_data.get('hrv_score'):
            message += f"❤️ *HRV Score:* {health_data['hrv_score']}\n"
        if health_data.get('body_battery'):
            message += f"🔋 *Body Battery:* {health_data['body_battery']}/100\n"
        
        # Activity metrics
        if health_data.get('steps'):
            message += f"👣 *Steps:* {health_data['steps']:,}\n"
        if health_data.get('resting_heart_rate'):
            message += f"💓 *Resting HR:* {health_data['resting_heart_rate']} bpm\n"
        
        # Stress removed per product decision

        # Optional: VO2max and Weight if present
        if health_data.get('vo2max_running'):
            try:
                message += f"🏃 *VO2max (Run):* {float(health_data['vo2max_running']):.1f} ml/kg/min\n"
            except Exception:
                pass
        if health_data.get('vo2max_cycling'):
            try:
                message += f"🚴 *VO2max (Bike):* {float(health_data['vo2max_cycling']):.1f} ml/kg/min\n"
            except Exception:
                pass
        bw = health_data.get('body_weight_kg')
        if isinstance(bw, (int, float)):
            # Normalize extreme values (likely grams) for display only
            bw_norm = float(bw)
            if bw_norm > 250:  # assume grams
                bw_norm = bw_norm / 1000.0
            message += f"⚖️ *Weight:* {bw_norm:.1f} kg\n"
        
        if len(message.split('\n')) <= 3:  # Only header
            message += "No detailed metrics available for today."
        
        return message

    def _format_today_from_dashboard(self, dash: Dict[str, Any]) -> str:
        lh = dash.get('latest_health') or {}
        date = lh.get('date', 'Unknown')
        msg = f"📊 *Health Data for {date}*\n\n"
        trends = dash.get('trends') or {}
        def _arrow(val: float | int | None, invert: bool = False) -> str:
            try:
                delta = float(val)
            except Exception:
                return ""
            if delta > 0:
                return " 🟢↑" if not invert else " 🔴↓"
            if delta < 0:
                return " 🔴↓" if not invert else " 🟢↑"
            return ""
        def _delta_for(key: str) -> float | None:
            # Try common layouts: trends['7d'][key]['delta'] or trends[key]['7d_delta'] etc.
            try:
                t7 = trends.get('7d') or {}
                k = t7.get(key)
                if isinstance(k, dict):
                    for cand in ('delta', 'change', 'diff'):
                        if isinstance(k.get(cand), (int, float)):
                            return float(k[cand])
                if isinstance(t7.get(f'{key}_delta'), (int, float)):
                    return float(t7.get(f'{key}_delta'))
            except Exception:
                pass
            try:
                k = trends.get(key) or {}
                for cand in ('delta_7d', 'change_7d', 'd7', 'week_delta'):
                    if isinstance(k.get(cand), (int, float)):
                        return float(k[cand])
            except Exception:
                pass
            return None
        # Sleep
        if lh.get('sleep_score') is not None:
            msg += f"😴 *Sleep Score:* {lh['sleep_score']}/100" + _arrow(_delta_for('sleep_score')) + "\n"
        if lh.get('sleep_hours'): msg += f"🕐 *Sleep Duration:* {self._format_hours_as_hhmm(lh['sleep_hours'])}\n"
        # Recovery
        if lh.get('hrv_score') is not None:
            msg += f"❤️ *HRV Score:* {lh['hrv_score']}" + _arrow(_delta_for('hrv_score')) + "\n"
        if lh.get('body_battery') is not None:
            msg += f"🔋 *Body Battery:* {lh['body_battery']}/100" + _arrow(_delta_for('body_battery')) + "\n"
        # Activity and HR
        if lh.get('steps') is not None:
            msg += f"👣 *Steps:* {lh['steps']:,}" + _arrow(_delta_for('steps')) + "\n"
        if lh.get('resting_heart_rate') is not None:
            msg += f"💓 *Resting HR:* {lh['resting_heart_rate']} bpm" + _arrow(_delta_for('resting_heart_rate'), invert=True) + "\n"
        # Stress removed per product decision
        # VO2 and Weight
        try:
            if lh.get('vo2max_running'):
                msg += f"🏃 *VO2max (Run):* {float(lh['vo2max_running']):.1f} ml/kg/min" + _arrow(_delta_for('vo2max_running')) + "\n"
            if lh.get('vo2max_cycling'):
                msg += f"🚴 *VO2max (Bike):* {float(lh['vo2max_cycling']):.1f} ml/kg/min" + _arrow(_delta_for('vo2max_cycling')) + "\n"
        except Exception:
            pass
        bw = lh.get('body_weight_kg')
        if isinstance(bw, (int, float)):
            bw_norm = float(bw)
            if bw_norm > 250:
                bw_norm = bw_norm / 1000.0
            msg += f"⚖️ *Weight:* {bw_norm:.1f} kg" + _arrow(_delta_for('body_weight_kg'), invert=True) + "\n"
        # If minimal data, add a short hint from trends if present
        trends = dash.get('trends') or {}
        if len(msg.split('\n')) <= 3 and isinstance(trends, dict):
            try:
                t30 = trends.get('30d') or {}
                tl = t30.get('training_load') or {}
                load_hint = None
                if tl.get('last_acute_7d') or tl.get('last_chronic_28d'):
                    load_hint = f"📈 Load 7d/28d: {tl.get('last_acute_7d','?')}/{tl.get('last_chronic_28d','?')}"
                acts = t30.get('activities') or {}
                act_hint = None
                if acts.get('count'):
                    act_hint = f"🏃 Activities 30d: {acts['count']}"
                hints = [h for h in [load_hint, act_hint] if h]
                if hints:
                    msg += "\n" + "\n".join(hints)
            except Exception:
                pass
        if len(msg.split('\n')) <= 3:
            msg += "No detailed metrics available for today."
        return msg

    def _format_hours_as_hhmm(self, hours_value: Any) -> str:
        """Convert decimal hours to HH:MM string (e.g., 7.5 -> 07:30)."""
        try:
            hours_float = float(hours_value)
        except (TypeError, ValueError):
            return "00:00"
        if hours_float < 0:
            hours_float = 0.0
        total_minutes = int(round(hours_float * 60))
        hours_part = total_minutes // 60
        minutes_part = total_minutes % 60
        return f"{hours_part:02d}:{minutes_part:02d}"
    
    def _format_activities(self, activities: List[Dict[str, Any]]) -> str:
        """Format activities into a readable message"""
        if not activities:
            return "🏃 No recent activities found."
        
        message = f"🏃 *Recent Activities ({len(activities)} found)*\n\n"
        
        for activity in activities[:5]:  # Show last 5
            name = activity.get('name', 'Unnamed Activity')
            activity_type = activity.get('activity_type', 'Activity')
            distance = activity.get('distance', 0)
            duration = activity.get('duration', 0)
            start_date = activity.get('start_date', '')
            
            # Format date
            if start_date:
                try:
                    date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%b %d')
                except:
                    date_str = start_date[:10]
            else:
                date_str = "Unknown"
            
            message += f"📅 *{date_str}* - {activity_type}\n"
            message += f"   🏃 {name}\n"
            
            if distance > 0:
                message += f"   📏 {distance:.2f} km"
            if duration > 0:
                message += f" ⏱️ {int(duration)} min"
            message += "\n\n"
        
        if len(activities) > 5:
            message += f"... and {len(activities) - 5} more activities\n"
        
        message += "Use `/activities` for full details!"
        
        return message
    
    async def _on_button_pressed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses for post-workout actions."""
        try:
            query = update.callback_query
            if not query:
                return
            data = (query.data or "").strip()
            await query.answer()
            if data == "log_workout":
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                await query.message.reply_text("✅ Logged today's workout (placeholder).")
            elif data == "request_different":
                await query.message.reply_text("Tell me what you'd prefer (e.g., 'easy gravel ride') or say 'Suggest something easier'.")
            elif data == "chat_coach":
                await query.message.reply_text("You can ask me anything—I'm here to help. For example: 'Why this workout today?'")
            elif data == "intent_daily":
                # Use dedicated daily-discussion endpoint to avoid hardcoded prompts
                await query.message.reply_text("🤖 Preparing today's recommendation...")
                try:
                    resp = requests.post(
                        f"{settings.agentic_api_base_url}/daily-discussion",
                        json={},
                        timeout=1800,
                    )
                    if resp.status_code == 200:
                        payload = resp.json() or {}
                        ans = payload.get("telegram_message", "")
                        if len(ans) > 3500:
                            ans = ans[:3500] + "..."
                        try:
                            await query.message.reply_text(
                                f"✅ Plan ready!\n\n{ans}",
                                parse_mode=ParseMode.MARKDOWN,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🔄 Request Different", callback_data="request_different")],
                                    [InlineKeyboardButton("💬 Chat with Coach", callback_data="chat_coach")],
                                ])
                            )
                        except Exception:
                            await query.message.reply_text(f"✅ Plan ready!\n\n{ans}")
                    else:
                        await query.message.reply_text("❌ Agentic API error. Try again later.")
                except Exception:
                    await query.message.reply_text("⚠️ Could not contact Agentic API (network).")
            elif data == "intent_change_workout":
                chat_id = str(query.message.chat_id)
                self._awaiting_workout_change[chat_id] = True
                await query.message.reply_text("What would you prefer for today? e.g., 'easy gravel ride' or 'short recovery run'.")
            # Removed buttons that send hardcoded prompts to the Agentic API
            # Goals/events inline flows disabled
            # events_refresh handled above
            # events_add_goal disabled
            # events_del disabled
            # events_chgdate disabled
        except Exception:
            pass

    def _build_coach_intents_keyboard(self) -> InlineKeyboardMarkup:
        # Minimal, no hardcoded prompts except daily trigger (handled server-side)
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🏋️ Workout of the Day", callback_data="intent_daily")],
            [InlineKeyboardButton("🔄 Request Different Workout", callback_data="intent_change_workout")],
            # Goals & Events temporarily removed from inline menu
        ])

    async def _route_and_reply(self, query, text: str, intent: str | None = None, metric: str | None = None):
        try:
            session_id = self._ensure_session_id(str(query.message.chat_id))
            payload = {"text": text, "session_id": session_id}
            if intent:
                payload["intent"] = intent
            if metric:
                payload["metric"] = metric
            if self._debug_mode.get(str(query.message.chat_id)):
                payload["debug"] = True
            resp = requests.post(f"{settings.agentic_api_base_url}/route", json=payload, timeout=1800)
            if resp.status_code == 200:
                jr = (resp.json() or {})
                ans = jr.get("reply", "")
                if self._debug_mode.get(str(query.message.chat_id)) and isinstance(jr.get("context_debug"), dict):
                    dbg = jr.get("context_debug")
                    extras = []
                    try:
                        extras.append(f"ctx keys: {len(dbg.get('latest_health_keys') or [])}")
                        extras.append(f"has trends: {bool(dbg.get('has_trends'))}")
                        extras.append(f"has baselines: {bool(dbg.get('has_baselines'))}")
                        extras.append(f"short mem turns: {len(dbg.get('short_memory') or [])}")
                    except Exception:
                        pass
                    if extras:
                        ans = (ans or "") + "\n\n_\[debug: " + ", ".join(extras) + "\]_"
                if ans:
                    try:
                        await query.message.reply_text(ans, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔄 Request Different", callback_data="request_different")],
                            [InlineKeyboardButton("💬 Chat with Coach", callback_data="chat_coach")],
                        ]))
                    except Exception:
                        await query.message.reply_text(ans)
                    return
        except Exception:
            pass
        await query.message.reply_text("⚠️ Could not reach the AI team right now.")

    # --------- Events helpers ---------
    # Event helpers disabled

    # Event helpers disabled

    # Event helpers disabled

    # Event rendering disabled
    
    # Quick Tips feature removed per product decision
    
    # Settings screen removed
    
    async def _handle_general_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general conversational messages via Agentic API project manager router."""
        user_msg = update.message.text.strip()
        try:
            chat_id = str(update.effective_chat.id) if update and update.effective_chat else None
            session_id = self._ensure_session_id(chat_id) if chat_id else None
            payload = {"text": user_msg, "session_id": session_id}
            if self._debug_mode.get(chat_id):
                payload["debug"] = True
            resp = requests.post(
                f"{settings.agentic_api_base_url}/route",
                json=payload,
                timeout=1800,
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("reply", "")
                if self._debug_mode.get(chat_id) and isinstance(data.get("context_debug"), dict):
                    dbg = data.get("context_debug")
                    extras = []
                    try:
                        extras.append(f"ctx keys: {len(dbg.get('latest_health_keys') or [])}")
                        extras.append(f"has trends: {bool(dbg.get('has_trends'))}")
                        extras.append(f"has baselines: {bool(dbg.get('has_baselines'))}")
                        extras.append(f"short mem turns: {len(dbg.get('short_memory') or [])}")
                    except Exception:
                        pass
                    if extras:
                        answer = (answer or "") + "\n\n_\[debug: " + ", ".join(extras) + "\]_"
                if not answer:
                    raise ValueError("Empty reply from Agentic API")
                
                # Send the successful AI response
                try:
                    await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    # Fallback to plain text if markdown fails
                    await update.message.reply_text(answer)
                return  # Exit early on success
            else:
                raise RuntimeError(f"Agentic API status {resp.status_code}")
        except Exception as e:
            # Include terse hint for debugging in logs, friendly fallback to user
            self.logger.error(f"Agentic API fallback due to error: {e}")
            answer = (
                "I couldn't reach the AI team just now. Try /data for today's metrics or /activities for workouts."
            )
            # Send fallback message and menu hint only on error
            try:
                await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(answer)
            try:
                await update.message.reply_text("Use the menu or /help for options.", reply_markup=self.main_menu_keyboard)
            except Exception:
                pass

    # Session watchdog removed
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors once (deduplicated)."""
        self.logger.error(f"Exception while handling an update: {context.error}")
        if update and hasattr(update, 'effective_chat'):
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="🤖 Oops! Something went wrong. Please try again later."
                )
            except Exception:
                pass

    # --------- Session helpers ---------
    def _ensure_session_id(self, chat_id: Optional[str]) -> Optional[str]:
        """Return an existing session id for the chat, or create a new one."""
        if not chat_id:
            return None
        sid = self._session_ids.get(chat_id)
        if sid:
            return sid
        try:
            import uuid
            sid = str(uuid.uuid4())
        except Exception:
            sid = f"session-{datetime.now().timestamp()}"
        self._session_ids[chat_id] = sid
        return sid

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle per-chat debug mode to include compact context info from Agentic API replies."""
        try:
            chat_id = str(update.effective_chat.id) if update and update.effective_chat else None
            if not chat_id:
                return
            current = bool(self._debug_mode.get(chat_id))
            self._debug_mode[chat_id] = not current
            state = "ON" if self._debug_mode[chat_id] else "OFF"
            await update.message.reply_text(f"🪪 Debug mode {state}. Subsequent replies will {'include' if self._debug_mode[chat_id] else 'not include'} context hints.")
        except Exception:
            pass

    async def _unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alias unsupported commands like /end-session to supported ones."""
        try:
            text = (getattr(update, 'message', None) and update.message.text) or ""
            if text.startswith("/end-session") or text.startswith("/end_session"):
                # Delegate to /end
                await self.end_command(update, context)
                return
            await update.message.reply_text("Unknown command. Try /end, /help, or just type your question.")
        except Exception:
            pass