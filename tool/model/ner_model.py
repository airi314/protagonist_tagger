from tool.gender_checker import get_personal_titles


class NERModel:
    def __init__(self, save_personal_titles, fix_personal_titles):
        self.save_personal_titles = save_personal_titles
        self.personal_titles = tuple(get_personal_titles())
        self.fix_personal_titles = fix_personal_titles
