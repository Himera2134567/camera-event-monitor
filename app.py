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
import time  # для тайминга

# --- Глобальный буфер для обработки детекций ---
# структура: { cam_id: { 'detections': [ {'timestamp': int, 'has': bool}, ... ],
#                       'last_signal': int } }
detection_buffers = {}

def process_detections_py(cam_id: int, has_detection: bool, current_ms: int) -> bool:
   
    time_window = 5000  # миллисекунд
    threshold = 60.0    # процент

    buf = detection_buffers.setdefault(cam_id, {'detections': [], 'last_signal': 0})

    # 1) Очищаем старые записи
    buf['detections'] = [
        d for d in buf['detections']
        if current_ms - d['timestamp'] <= time_window
    ]

    # 2) Добавляем текущую запись
    buf['detections'].append({'timestamp': current_ms, 'has': has_detection})

    dets = buf['detections']
    if len(dets) < 2:
        return False

    earliest = dets[0]['timestamp']
    latest   = dets[-1]['timestamp']
    total    = latest - earliest
    if total < time_window * 0.6:
        return False

    # 3) Считаем время наличия детекции
    det_time = 0
    for prev, curr in zip(dets, dets[1:]):
        if prev['has']:
            det_time += (curr['timestamp'] - prev['timestamp'])
    percent = (det_time / total) * 100

    # 4) Если процент достаточный и не было сигнала в последние 5с
    if percent >= threshold and (current_ms - buf['last_signal'] >= time_window):
        buf['last_signal'] = current_ms
        buf['detections'].clear()
        return True

    return False


# --- Файлы для хранения данных ---

# --- SMTP-настройки для bk.ru ---

# --- Утилиты для пользователей ---
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False)

# --- Утилиты для событий ---
def load_events():
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_events(events):
    with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False)

def add_event(frame, cam_id, conf):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(EVENTS_DIR, exist_ok=True)
    path = os.path.join(EVENTS_DIR, f'{ts}.jpg')
    cv2.imwrite(path, frame)
    ev = {
        'time': ts,
        'file': path,
        'cam': cam_id,
        'conf': conf,
        'read': False,
        'reaction': None
    }
    events = load_events()
    events.insert(0, ev)
    save_events(events)

# --- Инициализация сессии ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = None

# --- Страницы ---
def page_register():
    st.title("Регистрация")
    users = load_users()
    login = st.text_input("Логин")
    pwd   = st.text_input("Пароль", type="password")
    if st.button("Зарегистрироваться"):
        if not login or not pwd:
            st.error("Введите логин и пароль")
        elif login in users:
            st.error("Пользователь уже существует")
        else:
            users[login] = hash_password(pwd)
            save_users(users)
            st.success("Успешно! Перейдите на вкладку «Вход» и авторизуйтесь.")
            st.session_state.page = "Вход"

def page_login():
    st.title("Вход")
    users = load_users()
    login = st.text_input("Логин")
    pwd   = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        if verify_password(pwd, users.get(login, "")):
            st.session_state.user = login
            st.success(f"Добро пожаловать, {login}!")
            st.session_state.page = "Мониторинг"
            st.rerun()
        else:
            st.error("Неверный логин или пароль")

def page_monitoring():
    st.title(f"Мониторинг (пользователь: {st.session_state.user})")
    run = st.checkbox("Запустить камеру")
    FRAME = st.empty()
    if not run:
        return

    if not MODEL_PATH.is_file():
        st.error("Модель не найдена. Положите best.pt в models рядом с EXE или задайте MODEL_PATH.")
        return
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(0)
    cnt = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Камера недоступна")
            break

        if cnt % 3 == 0:
            res = model(frame, conf=0.6)[0]
            boxes = res.boxes
            has_det = boxes is not None and len(boxes) > 0

            now_ms = int(time.time() * 1000)
            if process_detections_py(cam_id=1, has_detection=has_det, current_ms=now_ms):
                conf = float(boxes[0].conf.cpu().numpy()) if has_det else 0.0
                add_event(frame, cam_id=1, conf=conf)

            annotated = res.plot()
        else:
            annotated = frame

        FRAME.image(annotated, channels='BGR')
        cnt += 1

    cap.release()

def page_events():
    st.title("События")

    # 1. Фильтрация несуществующих файлов
    events = load_events()
    events = [e for e in events if os.path.exists(e['file'])]
    save_events(events)

    # 2. Удалить всё
    if st.button("🗑️ Удалить все события"):
        for e in events:
            try: os.remove(e['file'])
            except: pass
        events.clear()
        save_events(events)
        st.success("Все события удалены")
        return

    # 3. Пометка новых и подсчёт
    new_count = sum(not e['read'] for e in events)
    for e in events:
        if not e['read']:
            e['read'] = True
    save_events(events)

    # 4. Основные табы
    tab_unread, tab_read = st.tabs([
        f"Непрочитанные ({new_count})",
        "Прочитанные"
    ])

    # Непрочитанные
    with tab_unread:
        to_process = [e for e in events if e['reaction'] is None]
        if not to_process:
            st.info("Нет новых событий.")
        for idx, ev in enumerate(to_process):
            header = f"{ev['time']} — Камера {ev['cam']} — conf={ev['conf']:.2f}"
            with st.expander(header, expanded=False):
                st.image(ev['file'], use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("✅", key=f"ok_{idx}"):
                    ev['reaction'] = 'green'; save_events(events); return
                if c2.button("⚠️", key=f"warn_{idx}"):
                    ev['reaction'] = 'yellow'; save_events(events); return
                if c3.button("❌", key=f"alert_{idx}"):
                    ev['reaction'] = 'red'; save_events(events); return
                if c4.button("Удалить", key=f"del_{idx}"):
                    try: os.remove(ev['file'])
                    except: pass
                    events = [e for e in events if e['time'] != ev['time']]
                    save_events(events)
                    return

    # Прочитанные
    with tab_read:
        red_events    = [e for e in events if e.get('reaction')=='red']
        yellow_events = [e for e in events if e.get('reaction')=='yellow']
        green_events  = [e for e in events if e.get('reaction')=='green']

        inner_tabs = st.tabs([
            f"🔴 Красные ({len(red_events)})",
            f"🟡 Жёлтые ({len(yellow_events)})",
            f"🟢 Зелёные ({len(green_events)})"
        ])

        for i, (label, lst, prefix) in enumerate([
            ("🔴 Красные", red_events, "red"),
            ("🟡 Жёлтые", yellow_events, "yellow"),
            ("🟢 Зелёные", green_events, "green"),
        ]):
            with inner_tabs[i]:
                if not lst:
                    st.info(f"Нет событий: {label.lower()}.")
                for idx, ev in enumerate(lst):
                    header = f"{ev['time']} — Камера {ev['cam']} — conf={ev['conf']:.2f}"
                    with st.expander(header, expanded=False):
                        st.image(ev['file'], use_container_width=True)
                        if st.button("Удалить", key=f"del_{prefix}_{idx}"):
                            try: os.remove(ev['file'])
                            except: pass
                            events = [e for e in events if e['time'] != ev['time']]
                            save_events(events)
                            return

def page_alert():
    st.title("Почтовое уведомление")
    if st.button("Отправить уведомление"):
        if not all((SMTP_SERVER, SMTP_USER, SMTP_PASS, SMTP_TO)):
            st.error("Заполните SMTP-настройки в .env рядом с EXE или через переменные окружения.")
            return
        try:
            msg = EmailMessage()
            msg["Subject"] = "Тревога! Камера 1"
            msg["From"]    = SMTP_USER
            msg["To"]      = SMTP_TO
            msg.set_content(
                "Охранник обнаружил подозрительную сумку на Камере 1.\n"
                "Просьба немедленно отреагировать!"
            )
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            st.success("Уведомление отправлено настроенному получателю.")
        except Exception as e:
            st.error(f"Ошибка отправки письма: {e}")

# --- Навигация ---
if st.session_state.user:
    menu = ["Мониторинг", "События", "Тревога", "Выйти"]
    if st.session_state.page not in menu:
        st.session_state.page = "Мониторинг"
    choice = st.sidebar.radio("Меню", menu, index=menu.index(st.session_state.page))
    st.session_state.page = choice

    if choice == "Мониторинг":
        page_monitoring()
    elif choice == "События":
        page_events()
    elif choice == "Тревога":
        page_alert()
    else:
        st.session_state.user = None
        st.session_state.page = None
        st.rerun()
else:
    menu = ["Регистрация", "Вход"]
    if st.session_state.page not in menu:
        st.session_state.page = "Вход"
    choice = st.sidebar.radio("Меню", menu, index=menu.index(st.session_state.page))
    st.session_state.page = choice

    if choice == "Регистрация":
        page_register()
    else:
        page_login()
