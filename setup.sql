-- MySQL table creation example

CREATE DATABASE sketch_db;
USE sketch_db;

CREATE TABLE images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sketch LONGBLOB NOT NULL,
    generated LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
