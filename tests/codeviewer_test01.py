def calculate_factors(n):
    """返回一个数的所有因子"""
    factors = [i for i in range(1, n + 1) if n % i == 0]
    return factors

def is_prime(num):
    """检查一个数是否为质!数"""
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_primes_in_range(start, end):
    """在给定范围内找到所有质数"""
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def main():
    number = 10
    print(f"Factors of {number}: {calculate_factors(number)}")

    start_range = 10
    end_range = 50
    primes = find_primes_in_range(start_range, end_range)
    print(f"Prime numbers between {start_range} and {end_range}: {primes}")

if __name__ == "__main__":
    main()
    # * Factors of 10: [1, 2, 5, 10]
    # TODO helloworld
    # ? blue
    # ! red 
    # // delete

    """
    ! red
    ? blue
    * green
    // delete
    TODO helloworld
    """