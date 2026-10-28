import streamlit as st
import cv2
from ultralytics import YOLO
import os
import datetime
import json
from settings import USERS_FILE, EVENTS_FILE, EVENTS_DIR, MODEL_PATH, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TO
from security import hash_password, verify_password
import smtplib
from email.message import EmailMessage

# ──────────────────────
#  Настройки и файлы
# ──────────────────────


# ══════════════════════
#  Утилиты
# ══════════════════════

def _safe_rerun():
    """Перезапуск скрипта для разных версий Streamlit без ошибок."""
    try:
        st.rerun()
    except AttributeError:
        st.rerun()

# ─── Работа с Pickle‑файлами ───

def _load(path, default):
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else default

def _save(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)

# ─── Пользователи ───

def load_users():
    return _load(USERS_FILE, {})

def save_users(d):
    _save(USERS_FILE, d)

# ─── События ───

def load_events():
    return _load(EVENTS_FILE, [])

def save_events(lst):
    _save(EVENTS_FILE, lst)


def add_event(img_bgr, cam_id: int, conf: float):
    """Сохраняем кадр с аннотациями и добавляем событие."""
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(EVENTS_DIR, f"{ts}.jpg")
    os.makedirs(EVENTS_DIR, exist_ok=True)
    cv2.imwrite(path, img_bgr)

    ev = dict(time=ts, file=path, cam=cam_id, conf=conf, read=False, reaction=None)
    ev_list = load_events()
    ev_list.insert(0, ev)
    save_events(ev_list)

# ══════════════════════
#  Инициализация сессии
# ══════════════════════

st.session_state.setdefault("user", None)
st.session_state.setdefault("page", None)

# ══════════════════════
#  Страницы
# ══════════════════════

def page_register():
    st.title("Регистрация")
    login = st.text_input("Логин")
    pwd   = st.text_input("Пароль", type="password")
    if st.button("Зарегистрироваться"):
        users = load_users()
        if not login or not pwd:
            st.error("Введите логин и пароль")
        elif login in users:
            st.error("Такой пользователь уже существует")
        else:
            users[login] = hash_password(pwd)
            save_users(users)
            st.session_state.user = login
            st.session_state.page = "Мониторинг"
            st.success("Регистрация успешна, выполняем вход…")
            _safe_rerun()


def page_login():
    st.title("Вход")
    login = st.text_input("Логин")
    pwd   = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        if verify_password(pwd, load_users().get(login, "")):
            st.session_state.user = login
            st.session_state.page = "Мониторинг"
            _safe_rerun()
        else:
            st.error("Неверный логин/пароль")


def page_monitoring():
    st.title(f"Мониторинг (пользователь: {st.session_state.user})")
    run_cam = st.checkbox("Запустить камеру")
    canvas  = st.empty()
    if not run_cam:
        return

    if not MODEL_PATH.is_file():
        st.error("Модель не найдена. Положите best.pt в models рядом с EXE или задайте MODEL_PATH.")
        return
    model = YOLO(str(MODEL_PATH))
    cap   = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Не удалось открыть камеру")
        return

    fid, THR = 0, 0.6
    while run_cam:
        ok, frame = cap.read()
        if not ok:
            st.warning("Нет сигнала от камеры")
            break

        annotated = frame
        if fid % 3 == 0:
            res = model(frame, conf=THR)[0]
            annotated = res.plot()  # Ultralytics выводит bbox + label
            if len(res.boxes) and float(res.boxes.conf.max()) > THR:
                add_event(annotated, cam_id=1, conf=float(res.boxes.conf.max()))
        canvas.image(annotated, channels="BGR")
        fid += 1

    cap.release()


def _event_button(label, key, action):
    """Кнопка без перезагрузки страницы."""
    if st.button(label, key=key):
        action()
        st.rerun()  # обновляем только текущую страницу


def page_events():
    st.title("События")
    ev = [e for e in load_events() if os.path.exists(e["file"])]
    save_events(ev)

    unread = [e for e in ev if e["reaction"] is None]
    read   = [e for e in ev if e["reaction"] is not None]

    t1, t2 = st.tabs([f"Непрочитанные ({len(unread)})", "Прочитанные"])

    # ─── Непрочитанные ───
    with t1:
        if not unread:
            st.info("Новых событий нет.")
        for idx, e in enumerate(unread):
            hdr = f"{e['time']} — Камера {e['cam']} — conf={e['conf']:.2f}"
            with st.expander(hdr):
                st.image(e['file'], use_container_width=True)
                cols = st.columns(4)
                _event_button("🟢", f"g{idx}",  lambda e=e: (e.update(reaction="green"), save_events(ev)))
                _event_button("🟡", f"y{idx}",  lambda e=e: (e.update(reaction="yellow"), save_events(ev)))
                _event_button("🔴", f"r{idx}",  lambda e=e: (e.update(reaction="red"), save_events(ev)))
                _event_button("Удалить", f"d{idx}", lambda e=e: (os.remove(e['file']), ev.remove(e), save_events(ev)))

    # ─── Прочитанные ───
    with t2:
        groups = {
            "🔴 Красные":  [e for e in read if e['reaction'] == 'red'],
            "🟡 Жёлтые":   [e for e in read if e['reaction'] == 'yellow'],
            "🟢 Зелёные":  [e for e in read if e['reaction'] == 'green'],
        }
        sub = st.tabs([f"{n} ({len(lst)})" for n, lst in groups.items()])
        for (name, lst), s in zip(groups.items(), sub):
            with s:
                if not lst:
                    st.info("Нет данных")
                for idx, e in enumerate(lst):
                    hdr = f"{e['time']} — Камера {e['cam']} — conf={e['conf']:.2f}"
                    with st.expander(hdr):
                        st.image(e['file'], use_container_width=True)
                        _event_button("Удалить", f"del_{name}_{idx}", lambda e=e: (os.remove(e['file']), ev.remove(e), save_events(ev)))


def page_alert():
    st.title("Почтовое уведомление")
    if st.button("Отправить уведомление"):
        if not all((SMTP_SERVER, SMTP_USER, SMTP_PASS, SMTP_TO)):
            st.error("Заполните SMTP-настройки в .env рядом с EXE или через переменные окружения.")
            return
        try:
            msg = EmailMessage()
            msg["Subject"] = "Тревога! Камера 1"
            msg["From"] = SMTP_USER
            msg["To"] = SMTP_TO
            msg.set_content("Обнаружен подозрительный объект. Требуется реагирование.")
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            st.success("Уведомление отправлено настроенному получателю.")
        except Exception as err:
            st.error(f"Ошибка отправки: {err}")

# ══════════════════════
#  Навигация
# ══════════════════════

if st.session_state.user:
    MENU = ["Мониторинг", "События", "Тревога", "Выйти"]
    st.session_state.page = st.session_state.page or "Мониторинг"
    choice = st.sidebar.radio("Меню", MENU, index=MENU.index(st.session_state.page))
    st.session_state.page = choice

    if choice == "Мониторинг":
        page_monitoring()
    elif choice == "События":
        page_events()
    elif choice == "Тревога":
        page_alert()
    else:  # Выход
        st.session_state.user = None
        st.session_state.page = None
        _safe_rerun()
else:
    MENU = ["Регистрация", "Вход"]
    st.session_state.page = st.session_state.page or "Вход"
    choice = st.sidebar.radio("Меню", MENU, index=MENU.index(st.session_state.page))
    st.session_state.page = choice

    if choice == "Регистрация":
        page_register()
    else:
        page_login()
