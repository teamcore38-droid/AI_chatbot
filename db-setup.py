from database import Database


def setup_database():
    db = Database()
    courses = db.list_courses()
    print("Database initialized successfully.")
    print("Courses available: {}".format(len(courses)))
    db.close()


if __name__ == "__main__":
    setup_database()

