from tool.gender_checker import get_personal_titles


class NERModel:
    def __init__(self, save_personal_titles):
        self.save_personal_titles = save_personal_titles
        if self.save_personal_titles:
            self.personal_titles = get_personal_titles()
