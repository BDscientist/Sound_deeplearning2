import pymysql
import datetime


def send_to_db(fl_name, spot_flag, higher_db):
    host = "rxgp1.iptime.org"
    port = 3380
    user = "liberabit"
    password = "liberabit!@"
    db_name = "MUTE-WEB"
    charset = "UTF8"

    conn = pymysql.connect(host=host, port=port, user=user, passwd=password, db=db_name, charset=charset)

    try:
        time = int(fl_name.split('_')[1])
        record_date = datetime.datetime.fromtimestamp(time / 1000).strftime('%Y-%m-%d %H:%M:%S')
        decibel = higher_db
        callsign = fl_name.split('_')[4].replace(".wav", "")
        with conn.cursor() as cursor:
            sql = "INSERT INTO `SPOT_DB` (`CREATE_DATE`, `UPDATE_DATE`, `HIGHER_DB`, `SPOT_FLAG`, `RECORD_DATE`, `CALLSIGN`) VALUES (now(), now(), %s, %s, %s, %s)"
            cursor.execute(sql, (decibel, spot_flag, record_date, callsign))
        conn.commit()
    except Exception as e:
        print(e)
        conn.rollback()
    finally:
        conn.close()


if __name__ == '__main__':
    file_name_empty = "Sound_1572243932423_Sound_1572243631121_.wav"
    file_name = "Sound_1572697892623_Sound_1572697591228_ESR226.wav"
    spot_flag = "PI1"
    db = 123.22
    send_to_db(file_name_empty, spot_flag, db)
