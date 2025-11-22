from filter.service import FilterService


class ChileScamFilter(FilterService):
    def __init__(self):
        patterns = [
            r"\bclave\b",
            r"\bcontraseñ?a\b",
            r"\bpassword\b",
            r"\bpwd\b",
            r"\btoken\b",
            r"\bclave\s*d[ií]namica\b",
            r"\bclave\s*secreta\b",
            # Bank-related
            r"\bbanco\b",
            r"\bsantander\b",
            r"\bscotiabank\b",
            r"\bestado\b",
            r"\bbci\b",
            r"\bsecurity\b",
            r"\bitau\b",
            # Phrases usually used in scams
            r"ejecutivo\s+del?\s+banco",
            r"verificar?\s+cuenta",
            r"actualizar\s+datos",
            r"bloque[oé]?\s+de\s+cuenta",
            r"validar\s+tu\s+identidad",
            r"transferencia\s+urgente",
            r"transferencia\s+inmediata",
            r"necesito\s+tu\s+c[oó]digo",
            r"c[oó]digo\s+de\s+verificaci[oó]n",
            r"clave\s+de\s+verificaci[oó]n",
            # Conversational Chilean scammy requests
            r"te\s+llamo\s+del?\s+banco",
            r"soy\s+del?\s+banco",
            r"necesito\s+confirmar\s+unos\s+datos",
            r"me\s+podr[ií]as\s+dar\s+tu\s+clave",
            r"p[aá]same\s+el\s+c[oó]digo",
            r"p[aá]same\s+tu\s+clave",
            r"p[aá]same\s+la\s+contraseñ?a",
        ]
        super().__init__(patterns)
