"""
main.py — نقطة دخول PeacockAgent المحدّثة
═══════════════════════════════════════════
يشغّل النظام الكامل:
  🔒 Security Layer
  🧠 Router / Orchestrator
  💾 Memory System
  🖥️ PyQt6 UI (لو موجود)

للتشغيل:
  python main.py

للتشغيل بدون UI (CLI mode):
  python main.py --cli
"""

import sys
import os
import argparse

# أضف المسار الحالي للـ imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli(api_key: str = ""):
    """تشغيل CLI بسيط للاختبار"""
    from agent import PeacockAgent

    print("═" * 55)
    print("  PeacockAgent v2.0 — CLI Mode")
    print("  Security + Router + Memory")
    print("═" * 55)
    print("  اكتب 'exit' للخروج | 'status' للحالة | 'clear' لمسح الجلسة")
    print("═" * 55 + "\n")

    agent = PeacockAgent(
        api_key       = api_key or os.getenv("GROQ_API_KEY", ""),
        verbose_logs  = True,
        log_file      = "data/security.log",
    )

    # Hook للتأكيد في CLI
    def cli_confirm(tool: str, risk, reason: str) -> bool:
        print(f"\n⚠️  عملية تحتاج موافقة:")
        print(f"   الأداة  : {tool}")
        print(f"   المخاطرة: {risk.value}")
        print(f"   السبب   : {reason}")
        ans = input("   هل توافق؟ (y/n): ").strip().lower()
        return ans in ("y", "yes", "نعم", "أيوه", "ايوه")

    agent.on_tool_confirm = cli_confirm

    while True:
        try:
            user_input = input("\n🦚 أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 مع السلامة!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("👋 مع السلامة!")
            break

        if user_input.lower() == "status":
            print(agent.diagnostics())
            continue

        if user_input.lower() == "clear":
            print(agent.clear_session())
            continue

        if user_input.lower().startswith("remember "):
            parts = user_input[9:].split("=", 1)
            if len(parts) == 2:
                print(agent.remember(parts[0].strip(), parts[1].strip()))
            continue

        if user_input.lower().startswith("recall "):
            key = user_input[7:].strip()
            print(f"🧠 {agent.recall(key)}")
            continue

        print("\n🦚 PeacockAgent: ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)


def run_ui(api_key: str = ""):
    """تشغيل واجهة PyQt6"""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QIcon
    except ImportError:
        print("❌ PyQt6 غير مثبت. شغّل: pip install PyQt6")
        print("   أو استخدم: python main.py --cli")
        sys.exit(1)

    try:
        from ui.app_window import PeacockAgentApp
    except ImportError:
        print("❌ ui/app_window.py غير موجود.")
        print("   تشغيل CLI mode بدلاً من ذلك...")
        run_cli(api_key)
        return

    from agent import PeacockAgent

    app = QApplication(sys.argv)
    app.setApplicationName("PeacockAgent")
    app.setApplicationVersion("2.0.0")

    # إنشاء الـ agent
    agent = PeacockAgent(
        api_key  = api_key or os.getenv("GROQ_API_KEY", ""),
        log_file = "data/security.log",
    )

    # تمرير الـ agent للـ UI
    window = PeacockAgentApp(agent=agent)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeacockAgent v2.0")
    parser.add_argument("--cli",     action="store_true", help="تشغيل CLI mode")
    parser.add_argument("--api-key", type=str, default="", help="Groq API Key")
    args = parser.parse_args()

    if args.cli:
        run_cli(args.api_key)
    else:
        run_ui(args.api_key)
