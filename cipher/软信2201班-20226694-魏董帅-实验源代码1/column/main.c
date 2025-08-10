#include <stdio.h>
#include <string.h>

// ���ܺ���
void encrypt(char *plaintext, int key[], int columns) {
    int length = strlen(plaintext);
    int rows = (length + columns - 1) / columns;
    char matrix[rows][columns];

    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (k < length) {
                matrix[i][j] = plaintext[k++];
            } else {
                matrix[i][j] = ' ';  // ���ո�
            }
        }
    }

    for (int j = 0; j < columns; j++) {
        int col = key[j];
        for (int i = 0; i < rows; i++) {
            printf("%c", matrix[i][col]);
        }
    }
}

// ���ܺ���
void decrypt(char *ciphertext, int key[], int columns) {
    int length = strlen(ciphertext);
    int rows = (length + columns - 1) / columns;
    char matrix[rows][columns];

    int k = 0;
    for (int j = 0; j < columns; j++) {
        int col = key[j];
        for (int i = 0; i < rows; i++) {
            matrix[i][col] = ciphertext[k++];
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%c", matrix[i][j]);
        }
    }
}

int main() {
    char plaintext[100];
    int key[] = {1, 0, 3, 2};  // ���û���˳��
    int columns = 4;

    printf("����������: ");
    fgets(plaintext, 100, stdin);
    plaintext[strcspn(plaintext, "\n")] = 0;  // ȥ��ĩβ�Ļ��з�
    printf("����: %s\n", plaintext);
    printf("���ܺ�: ");
    encrypt(plaintext, key, columns);

    char ciphertext[100];
    printf("\n\n����������: ");
    fgets(ciphertext, 100, stdin);
    ciphertext[strcspn(ciphertext, "\n")] = 0;  // ȥ��ĩβ�Ļ��з�
    printf("���ܺ�: ");
    decrypt(ciphertext, key, columns);

    return 0;
}
