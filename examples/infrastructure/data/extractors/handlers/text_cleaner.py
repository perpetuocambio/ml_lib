class TextCleaner:
    """Limpiador de texto extraído."""

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normaliza espacios en blanco en el texto."""
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Eliminar espacios al inicio y final
            line = line.strip()
            # Reemplazar múltiples espacios por uno solo
            line = " ".join(line.split())
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """Elimina líneas vacías múltiples."""
        if not text:
            return ""

        lines = text.split("\n")
        result = []
        prev_empty = False

        for line in lines:
            if line.strip() == "":
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return "\n".join(result)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpia el texto extraído eliminando caracteres innecesarios.

        Args:
            text: Texto a limpiar

        Returns:
            str: Texto limpio
        """
        if not text:
            return ""

        # Aplicar limpieza de espacios en blanco
        cleaned = TextCleaner.clean_whitespace(text)

        # Eliminar líneas vacías múltiples
        cleaned = TextCleaner.remove_empty_lines(cleaned)

        return cleaned
