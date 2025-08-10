#include <stdio.h>
#include <string.h>

// 加密函数
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
                matrix[i][j] = ' ';  // 填充空格
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

// 解密函数
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
    int key[] = {1, 0, 3, 2};  // 列置换的顺序
    int columns = 4;

    printf("请输入明文: ");
    fgets(plaintext, 100, stdin);
    plaintext[strcspn(plaintext, "\n")] = 0;  // 去除末尾的换行符
    printf("明文: %s\n", plaintext);
    printf("加密后: ");
    encrypt(plaintext, key, columns);

    char ciphertext[100];
    printf("\n\n请输入密文: ");
    fgets(ciphertext, 100, stdin);
    ciphertext[strcspn(ciphertext, "\n")] = 0;  // 去除末尾的换行符
    printf("解密后: ");
    decrypt(ciphertext, key, columns);

    return 0;
}
