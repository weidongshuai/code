#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 使用密钥对文本进行加密
char* vigenere_encrypt(char* text, char* key) {
    int textLen = strlen(text);
    int keyLen = strlen(key);
    char* encrypted = (char*)malloc(textLen + 1);
    int i;
    for (i = 0; i < textLen; i++) {
        // 对每个字符进行加密
        encrypted[i] = ((text[i] - 'A' + key[i % keyLen] - 'A') % 26) + 'A';
    }
    encrypted[i] = '\0';
    return encrypted;
}

// 使用密钥对文本进行解密
char* vigenere_decrypt(char* text, char* key) {
    int textLen = strlen(text);
    int keyLen = strlen(key);
    char* decrypted = (char*)malloc(textLen + 1);
    int i;
    for (i = 0; i < textLen; i++) {
        // 对每个字符进行解密
        decrypted[i] = ((text[i] - 'A' - (key[i % keyLen] - 'A') + 26) % 26) + 'A';
    }
    decrypted[i] = '\0';
    return decrypted;
}

int main() {
    char text[100]; // 明文
    char key[100]; // 密钥

    printf("请输入要加密的文本(大写): ");
    scanf("%s", text);

    printf("请输入密钥: ");
    scanf("%s", key);

    // 加密文本
    char* encryptedText = vigenere_encrypt(text, key);
    printf("加密后的文本: %s\n", encryptedText);

    // 解密文本
    char* decryptedText = vigenere_decrypt(encryptedText, key);
    printf("解密后的文本: %s\n", decryptedText);

    free(encryptedText);
    free(decryptedText);

    return 0;
}
