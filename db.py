import psycopg2 as psycopg2

# Make connection with db
def get_connection():
    connection = psycopg2.connect(user = "postgres",
                                  password = "postgres",
                                  host = "localhost",
                                  port = "5432",
                                  database = "postgres")
    return connection

# Close connection with db
def close_connection(connection):
    if connection:
        connection.close()
        print("Postgres connection is closed")