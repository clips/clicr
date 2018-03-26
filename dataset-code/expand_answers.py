try:
    import pymysql
except ImportError:
    print("Unable to import pymysql")


def conn():
    return pymysql.connect(host='localhost',
                           user="root",
                           password="",
                           db="umls",
                           cursorclass=pymysql.cursors.DictCursor)


def expand(cui, cur, downcase=False):
    def tolower(w):
        return w.lower() if downcase else w

    cur.execute("SELECT STR FROM MRCONSO WHERE CUI='{}' AND LAT='ENG'".format(cui))
    expanded_set = {tolower(i["STR"]) for i in cur.fetchall()}

    return expanded_set


if __name__ == "__main__":
    # connect to UMLS
    cur = conn().cursor()

    cui = "C0022660"  # c = "acute renal failure"
    print(cui)
    print(expand(cui, cur, downcase=True))
