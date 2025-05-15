def fibonacci(n, memo={}):
    if n in memo:  # Eğer sonuç cache'de varsa, doğrudan döndür
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)  # Sonucu hesapla ve sakla
    return memo[n]

print(fibonacci(50))  # Çok daha hızlı hesaplanır
