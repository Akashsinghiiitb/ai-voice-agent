# voice_agent/localization.py

LOCALIZATION_CONFIGS = {
    "default": {
        "name": "UIIC Health Insurance Bot",
        "sector": "Health Insurance",
        "languages": ["English"],
        "tts_lang": "en",
        "system_prompt": (
            "You are a strict, grounded Health Insurance assistant representing UIIC. "
            "Synthesize an answer for the user query based ONLY on the provided context. "
            "If the context does not contain enough information to resolve the question, "
            "reply exactly with: 'I don't have enough information in the knowledge base.'\n"
            "Do not make assumptions, do not use outside knowledge, and do not hallucinate."
        ),
        "objection_keywords": [
            "expensive", "don't need", "dont need", "no need", "waste of money",
            "why should i buy", "too high", "costly"
        ],
        "escalation_keywords": [
            "human", "representative", "support", "complaint", 
            "connect", "operator", "speak to someone", "customer service"
        ],
        "farewell_keywords": [
            "goodbye", "bye", "exit", "stop", "end conversation"
        ],
        "objection_response": "I understand your concern. Health insurance helps protect against unexpected medical expenses. I can explain the coverage benefits.",
        "escalation_response": "I will connect you with a customer support representative.",
        "farewell_response": "Goodbye! Thank you for contacting UIIC Health Insurance support. Have a wonderful day!",
        "seed_data": []  # Default data is already crawled/ingested in DB
    },
    "philippines": {
        "name": "Pioneer Life Insurance Bot",
        "sector": "Life Insurance",
        "languages": ["English", "Filipino (Tagalog)", "Taglish"],
        "tts_lang": "tl",
        "system_prompt": (
            "You are a Pioneer Life Insurance assistant in the Philippines. "
            "Speak naturally in Taglish (a conversational blend of English and Tagalog/Filipino). "
            "Be professional, warm, and helpful. "
            "You must integrate natural life insurance terminology in English, such as: "
            "premium, policy, beneficiary, rider, coverage, lapse, and bank referral.\n"
            "Use the provided context to answer questions. If the context does not contain the answer, "
            "use your default Pioneer Life knowledge politely in Taglish to guide the user. "
            "Focus on: premium reminder, renewal reminder, and lead qualification. "
            "Keep responses very brief, friendly, and conversational (under 3 sentences) for a phone call."
        ),
        "objection_keywords": [
            "mahal", "expensive", "hindi ko kailangan", "no need", "dont need", "don't need",
            "sayang pera", "waste of money", "bakit ko kailangan", "too high", "costly",
            "wala akong pera", "no budget", "mag-lapse", "bawas", "walang silbi"
        ],
        "escalation_keywords": [
            "tao", "human", "representative", "operator", "kausap", "makausap", "agent",
            "supervisor", "manager", "support", "customer service", "irefer", "i-transfer"
        ],
        "farewell_keywords": [
            "paalam", "goodbye", "bye", "exit", "stop", "tapos na", "salamat", "thank you", "okay na"
        ],
        "objection_response": "Naiintindihan ko po ang inyong concern. Mahalaga ang life insurance para sa future ng inyong beneficiaries. Pwede po nating pag-usapan ang lower coverage o riders na akma sa inyong budget.",
        "escalation_response": "Iho-hold ko po ang call at ikokonekta ko kayo sa aming customer service representative. Sandali lang po.",
        "farewell_response": "Maraming salamat sa pagtawag sa Pioneer Life! Mag-ingat po kayo palagi at magandang araw!",
        "seed_data": [
            {
                "title": "Pioneer Life Insurance Policy Guidelines",
                "content": "Premium reminders are sent automatically 30 days before the payment due date. If unpaid, the policy enters a 31-day grace period before it lapses.",
                "category": "Policy PDF",
                "source": "pioneer_life_terms.pdf",
                "page": "1",
                "section": "Premium Payments",
                "url": "https://pioneer.com.ph/life-insurance"
            },
            {
                "title": "Pioneer Life Insurance Policy Guidelines",
                "content": "A beneficiary is the designated person or entity who will receive the policy coverage benefits upon the death of the insured. You can update your beneficiary designations at any time.",
                "category": "Policy PDF",
                "source": "pioneer_life_terms.pdf",
                "page": "2",
                "section": "Beneficiaries",
                "url": "https://pioneer.com.ph/life-insurance"
            },
            {
                "title": "Pioneer Life Insurance Policy Guidelines",
                "content": "Riders are optional add-ons to a basic life insurance policy that provide extra coverage, such as critical illness rider, accidental death benefit rider, and premium waiver rider.",
                "category": "Policy PDF",
                "source": "pioneer_life_terms.pdf",
                "page": "3",
                "section": "Riders & Add-ons",
                "url": "https://pioneer.com.ph/life-insurance"
            },
            {
                "title": "Pioneer Life Insurance Policy Guidelines",
                "content": "A lapsed policy has expired because premium payments were not made within the grace period. Lapsed policies can be reinstated within 3 years by paying back premiums plus interest and proving insurability.",
                "category": "Policy PDF",
                "source": "pioneer_life_terms.pdf",
                "page": "4",
                "section": "Lapse & Reinstatement",
                "url": "https://pioneer.com.ph/life-insurance"
            },
            {
                "title": "Pioneer Life Insurance Policy Guidelines",
                "content": "We offer bank referral programs (bancassurance) where clients referred by our partner banks receive special premium rates and fast-track policy approval.",
                "category": "Policy PDF",
                "source": "pioneer_life_terms.pdf",
                "page": "5",
                "section": "Bank Referrals",
                "url": "https://pioneer.com.ph/life-insurance"
            }
        ]
    },
    "indonesia": {
        "name": "Adira Finance Consumer Finance Bot",
        "sector": "Consumer Finance",
        "languages": ["Formal Bahasa Indonesia", "Colloquial Bahasa", "Finance English loan words"],
        "tts_lang": "id",
        "system_prompt": (
            "You are an Adira Finance Consumer Finance voice assistant in Indonesia. "
            "Speak in a natural blend of Formal and Colloquial Bahasa Indonesia, integrating common finance English loan words. "
            "Be polite, professional, and clear. "
            "You must integrate local finance terminology: cicilan, tenor, denda, DP (Down Payment), jatuh tempo, angsuran, and pembiayaan.\n"
            "Use the provided context to answer questions. If the context does not contain the answer, "
            "reply politely in Bahasa using your Adira Finance knowledge. "
            "Focus on: installment reminder, loan follow-up, and customer qualification. "
            "Keep responses very brief, direct, and conversational (under 3 sentences) for a phone call."
        ),
        "objection_keywords": [
            "mahal", "expensive", "tidak butuh", "tidak perlu", "no need", "dont need",
            "buang uang", "waste of money", "kenapa harus", "too high", "costly",
            "tidak ada uang", "tidak sanggup", "bokek", "denda kemahalan", "keberatan"
        ],
        "escalation_keywords": [
            "manusia", "human", "representative", "operator", "hubungkan", "bicara", "staf",
            "petugas", "supervisor", "cs", "customer service", "agent", "sales", "telpon orang"
        ],
        "farewell_keywords": [
            "selamat tinggal", "goodbye", "bye", "exit", "stop", "selesai", "sudah",
            "terima kasih", "makasih", "dah", "oke cukup"
        ],
        "objection_response": "Saya memahami kesulitan Bapak atau Ibu. Pembiayaan kami dirancang untuk membantu cicilan kendaraan tetap terjangkau. Mari kita diskusikan tenor dan angsuran yang lebih cocok.",
        "escalation_response": "Baik, saya akan segera menghubungkan Anda dengan staf customer service supervisor kami. Mohon ditunggu sebentar.",
        "farewell_response": "Terima kasih telah menghubungi Adira Finance. Semoga sehat selalu dan selamat beraktivitas!",
        "seed_data": [
            {
                "title": "Adira Finance Terms of Service",
                "content": "Cicilan (installment) dan angsuran bulanan harus dibayarkan paling lambat pada tanggal jatuh tempo (due date) yang tercantum di kontrak pembiayaan.",
                "category": "Policy PDF",
                "source": "adira_finance_terms.pdf",
                "page": "1",
                "section": "Installments & Overdue",
                "url": "https://adira.co.id/consumer-finance"
            },
            {
                "title": "Adira Finance Terms of Service",
                "content": "Tenor adalah jangka waktu pelunasan pembiayaan. Kami menyediakan opsi tenor mulai dari 12 bulan, 24 bulan, hingga 48 bulan sesuai kesepakatan.",
                "category": "Policy PDF",
                "source": "adira_finance_terms.pdf",
                "page": "2",
                "section": "Loan Tenor",
                "url": "https://adira.co.id/consumer-finance"
            },
            {
                "title": "Adira Finance Terms of Service",
                "content": "Denda keterlambatan akan dikenakan sebesar 0.1% per hari dari jumlah angsuran yang terlambat, dihitung mulai H+1 dari tanggal jatuh tempo.",
                "category": "Policy PDF",
                "source": "adira_finance_terms.pdf",
                "page": "3",
                "section": "Late Penalty (Denda)",
                "url": "https://adira.co.id/consumer-finance"
            },
            {
                "title": "Adira Finance Terms of Service",
                "content": "DP (Down Payment) minimal untuk pembiayaan mobil adalah 15%, sedangkan untuk sepeda motor minimal adalah 10% dari harga kendaraan.",
                "category": "Policy PDF",
                "source": "adira_finance_terms.pdf",
                "page": "4",
                "section": "Down Payment (DP)",
                "url": "https://adira.co.id/consumer-finance"
            },
            {
                "title": "Adira Finance Terms of Service",
                "content": "Pembiayaan konsumen kami menyediakan dana cepat untuk pembelian kendaraan bermotor, elektronik, maupun pinjaman multiguna dengan jaminan BPKB.",
                "category": "Policy PDF",
                "source": "adira_finance_terms.pdf",
                "page": "5",
                "section": "Pembiayaan Overview",
                "url": "https://adira.co.id/consumer-finance"
            }
        ]
    }
}
